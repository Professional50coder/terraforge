"""Retrieval-augmented explanations from a free LLM, with a deterministic fallback.

Pipeline: model prediction + spectral statistics + the k most similar labelled patches
(from the vector index) are assembled into a grounded prompt. The LLM only *phrases*
facts we computed; it is told not to invent any. If no LLM is reachable the same facts
are rendered by a template, so the product never depends on a third party being up.

Backends (all free): Hugging Face Inference API (HF_TOKEN env var, free tier), a local
Ollama server, or the template fallback. Only synthetic/public EO statistics are sent.
"""
from __future__ import annotations

import json
import os
import urllib.request

SYSTEM = ("You are a remote-sensing analyst. Using ONLY the facts provided, write 2-3 "
          "sentences explaining the model's prediction. Do not add facts, numbers or "
          "locations that are not in the input. State uncertainty when confidence is low.")


def build_facts(label: str, confidence: float, ndvi: float | None, neighbours: list[dict]) -> dict:
    votes: dict[str, int] = {}
    for n in neighbours:
        votes[n["class"]] = votes.get(n["class"], 0) + 1
    return {
        "prediction": label, "confidence": round(confidence, 3),
        "mean_ndvi": None if ndvi is None else round(ndvi, 3),
        "similar_patch_classes": votes,
        "neighbours_agree": bool(votes) and max(votes.values()) / len(neighbours) >= 0.6,
    }


def template_explanation(f: dict) -> str:
    s = f"Predicted {f['prediction']} with {f['confidence']:.0%} confidence."
    if f["mean_ndvi"] is not None:
        level = "high" if f["mean_ndvi"] > 0.5 else "moderate" if f["mean_ndvi"] > 0.2 else "low"
        s += f" Mean NDVI is {f['mean_ndvi']} ({level} vegetation signal)."
    if f["similar_patch_classes"]:
        top = max(f["similar_patch_classes"], key=f["similar_patch_classes"].get)
        s += (f" The most similar reference patches are mostly {top}"
              f"{', consistent with' if f['neighbours_agree'] else ', which only partly supports'}"
              " the prediction.")
    if f["confidence"] < 0.6:
        s += " Confidence is low; treat this result as uncertain."
    return s


def load_env(path: str, allow: tuple[str, ...] | None = None) -> int:
    """Load KEY=VALUE lines from a local .env into os.environ (existing vars win).

    `allow` restricts which keys are imported (least privilege): a shared .env often holds
    unrelated credentials this process has no business seeing. Values are never printed or
    logged. Keeping secrets in a file outside the repo is what keeps them out of git.
    """
    n = 0
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if allow is not None and k not in allow:
            continue
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


# OpenAI-compatible chat endpoints: (url, env var holding the key, default model)
OPENAI_COMPATIBLE = {
    "hf": ("https://router.huggingface.co/v1/chat/completions", "HF_TOKEN",
           "HuggingFaceTB/SmolLM3-3B"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY",
             "openai/gpt-oss-20b"),
}


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    # Some API gateways reject urllib's default agent (HTTP 403, Cloudflare 1010).
    headers = {"User-Agent": "terraforge/0.1 (+https://github.com/Professional50coder/terraforge)",
               **headers}
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Explainer:
    def __init__(self, backend: str = "template", model: str | None = None, timeout: float = 20):
        if backend not in ("template", "ollama", *OPENAI_COMPATIBLE):
            raise ValueError(f"unknown backend {backend}")
        self.backend, self.timeout = backend, timeout
        defaults = {k: v[2] for k, v in OPENAI_COMPATIBLE.items()}
        self.model = model or {**defaults, "ollama": "llama3.2", "template": ""}[backend]

    def _llm(self, facts: dict) -> str:
        user = "Facts: " + json.dumps(facts)
        if self.backend in OPENAI_COMPATIBLE:
            url, env_var, _ = OPENAI_COMPATIBLE[self.backend]
            token = os.environ.get(env_var)
            if not token:
                raise RuntimeError(f"{env_var} not set")
            out = _post(url,
                        {"model": self.model, "max_tokens": 500, "temperature": 0.2,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": user}]},
                        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                        self.timeout)
            return out["choices"][0]["message"]["content"].strip()
        out = _post("http://127.0.0.1:11434/api/chat",
                    {"model": self.model, "stream": False, "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}]},
                    {"Content-Type": "application/json"}, self.timeout)
        return out["message"]["content"].strip()

    def explain(self, label, confidence, ndvi, neighbours) -> dict:
        facts = build_facts(label, confidence, ndvi, neighbours)
        if self.backend != "template":
            try:
                return {"text": self._llm(facts), "source": self.backend, "facts": facts}
            except Exception as e:  # network, auth, rate limit: degrade, never fail the request
                return {"text": template_explanation(facts), "source": "template",
                        "facts": facts, "fallback_reason": type(e).__name__}
        return {"text": template_explanation(facts), "source": "template", "facts": facts}
