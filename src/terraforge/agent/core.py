"""Tool-calling agent with guardrails.

Loop: send messages + tool schemas to an LLM; if it asks for tools, run them (validated,
exception-contained) and feed results back; stop when it answers or after `max_steps`.

Guardrails, and why each exists:
- Step cap: a confused model can loop on tools forever; the cap bounds cost and latency.
- Argument validation: LLMs emit malformed arguments; errors go back to the model as
  tool results so it can self-correct instead of crashing the request.
- Exception containment: a failing tool (e.g. network) becomes a readable error, never a 500.
- Grounding prompt: the model may only state facts returned by tools.
- Rule-based fallback: if no LLM is reachable the product still answers common questions.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from terraforge.agent.tools import Tool, build_tools

SYSTEM = ("You are TerraForge, a concise Earth-observation assistant. Use the tools to get "
          "facts. State ONLY what tool results contain; if a tool fails or data is missing, "
          "say so plainly. Do not invent coordinates, dates, numbers or imagery. Keep answers "
          "short and in plain language.")


DEFAULT_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")


class GroqChat:
    """OpenAI-compatible chat call against Groq's free endpoint (key from GROQ_API_KEY).

    Hosted model catalogues change (names get retired), so a 404 `model_not_found` moves on
    to the next candidate instead of failing. The first that works is remembered.
    Override with TERRAFORGE_LLM_MODEL.
    """
    def __init__(self, models: tuple[str, ...] | None = None, timeout: float = 25):
        env = os.environ.get("TERRAFORGE_LLM_MODEL")
        self.models = list(models or ((env,) if env else DEFAULT_MODELS))
        self.timeout = timeout

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        last: Exception | None = None
        for model in list(self.models):
            body = json.dumps({"model": model, "messages": messages, "tools": tools,
                               "tool_choice": "auto", "temperature": 0.1,
                               "max_tokens": 600}).encode()
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions", body,
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "terraforge/0.1 (+https://github.com/Professional50coder/terraforge)"},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.models.remove(model)
                    self.models.insert(0, model)  # remember the one that works
                    return json.loads(r.read())["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code != 404:  # only a missing model is worth trying the next one for
                    raise
        raise last or RuntimeError("no model available")


class Agent:
    def __init__(self, llm=None, tools: dict[str, Tool] | None = None, max_steps: int = 4):
        self.llm, self.tools, self.max_steps = llm, tools or build_tools(), max_steps

    def _run_tool(self, name: str, raw_args) -> dict:
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"unknown tool '{name}'", "available": list(self.tools)}
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            return {"result": tool.fn(**tool.validate(args))}
        except Exception as e:  # contained on purpose: tool failure must not kill the request
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, message: str, history: list[dict] | None = None) -> dict:
        trace: list[dict] = []
        if self.llm is None:
            return self._fallback(message, trace, reason="no_llm")
        msgs = [{"role": "system", "content": SYSTEM}, *(history or []),
                {"role": "user", "content": message}]
        schemas = [t.schema() for t in self.tools.values()]
        for _ in range(self.max_steps):
            try:
                reply = self.llm(msgs, schemas)
            except Exception as e:
                return self._fallback(message, trace, reason=type(e).__name__)
            calls = reply.get("tool_calls") or []
            if not calls:
                return {"answer": (reply.get("content") or "").strip(), "tools": trace,
                        "source": "llm"}
            msgs.append({"role": "assistant", "content": reply.get("content"), "tool_calls": calls})
            for c in calls:
                fn = c["function"]
                out = self._run_tool(fn["name"], fn.get("arguments"))
                trace.append({"tool": fn["name"], "arguments": fn.get("arguments"), **out})
                msgs.append({"role": "tool", "tool_call_id": c.get("id", fn["name"]),
                             "content": json.dumps(out)})
        return {"answer": "I could not finish that within the step limit. Try a narrower question.",
                "tools": trace, "source": "step_limit"}

    # -- deterministic fallback ------------------------------------------------------
    def _fallback(self, message: str, trace: list, reason: str) -> dict:
        m = message.lower()
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", message)]
        if ("daylight" in m or "day or night" in m or "night" in m) and len(nums) >= 2:
            out = self._run_tool("daylight_at", {"lat": nums[0], "lon": nums[1]})
            trace.append({"tool": "daylight_at", **out})
            if "result" in out:
                r = out["result"]
                ans = f"It is {'daylight' if r['daylight'] else 'night'} at {r['lat']}, {r['lon']} right now."
            else:
                ans = f"I could not check that: {out['error']}"
        elif "moon" in m or "sun" in m:
            out = self._run_tool("sky_now", {})
            trace.append({"tool": "sky_now", **out})
            r = out["result"]
            ans = (f"The Moon is {r['moon']['phase']} ({r['moon']['illumination']:.0%} lit). "
                   f"The Sun is over latitude {r['sun']['lat']:.1f}, longitude {r['sun']['lon']:.1f}.")
        else:
            ans = ("I can answer questions about day/night at a place (give latitude and longitude), "
                   "the Sun and Moon, areas, and Sentinel-2 scene search.")
        return {"answer": ans, "tools": trace, "source": "fallback", "fallback_reason": reason}
