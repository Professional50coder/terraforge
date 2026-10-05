import json

from terraforge.agent.core import Agent
from terraforge.agent.tools import build_tools


def call(name, args, id="c1"):
    return {"id": id, "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    """Fake LLM replaying a fixed list of assistant messages; records what it was sent."""
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, tools):
        self.seen.append(messages)
        return self.replies.pop(0)


def test_tool_call_then_answer_uses_real_tool_result():
    llm = Scripted([{"tool_calls": [call("daylight_at", {"lat": 0, "lon": 0,
                                                         "time": "2024-03-20T12:00:00Z"})]},
                    {"content": "It is daylight there."}])
    out = Agent(llm).run("is it day at 0,0?")
    assert out["answer"] == "It is daylight there." and out["source"] == "llm"
    assert out["tools"][0]["result"]["daylight"] is True
    tool_msg = [m for m in llm.seen[1] if m["role"] == "tool"][0]
    assert '"daylight": true' in tool_msg["content"]  # result was fed back to the model


def test_unknown_tool_and_bad_args_become_errors_not_crashes():
    llm = Scripted([{"tool_calls": [call("launch_rocket", {}), call("daylight_at", {"lat": "x", "lon": 0})]},
                    {"content": "Sorry, I couldn't do that."}])
    out = Agent(llm).run("hi")
    assert "unknown tool" in out["tools"][0]["error"]
    assert "must be a number" in out["tools"][1]["error"]
    assert out["answer"].startswith("Sorry")


def test_out_of_range_and_unknown_argument_rejected():
    t = build_tools()["daylight_at"]
    a = Agent(Scripted([]))
    assert "error" in a._run_tool("daylight_at", {"lat": 95, "lon": 0})
    assert "unknown argument" in a._run_tool("daylight_at", {"lat": 1, "lon": 1, "x": 2})["error"]
    assert t.name == "daylight_at"


def test_step_limit_stops_runaway_tool_loop():
    loop = {"tool_calls": [call("sky_now", {})]}
    out = Agent(Scripted([loop] * 10), max_steps=3).run("loop")
    assert out["source"] == "step_limit" and len(out["tools"]) == 3


def test_search_scenes_uses_injected_client_and_contains_network_errors():
    class Item:
        id, datetime, properties = "S2_x", "2024-06-01T10:00:00Z", {"eo:cloud_cover": 3.0}

    class Client:
        def search(self, q):
            return [Item()]

    ok = Agent(None, build_tools(Client()))._run_tool(
        "search_scenes", {"bbox": [5.9, 49.4, 6.5, 50.2], "start": "2024-06-01", "end": "2024-06-30"})
    assert ok["result"]["scenes"][0]["cloud_pct"] == 3.0

    class Down:
        def search(self, q):
            raise ConnectionError("no network")

    bad = Agent(None, build_tools(Down()))._run_tool(
        "search_scenes", {"bbox": [5.9, 49.4, 6.5, 50.2], "start": "2024-06-01", "end": "2024-06-30"})
    assert "ConnectionError" in bad["error"]


def test_llm_failure_falls_back_to_rules():
    def broken(m, t):
        raise TimeoutError()

    out = Agent(broken).run("Is it daylight at 48.8, 2.3 right now?")
    assert out["source"] == "fallback" and out["fallback_reason"] == "TimeoutError"
    assert "daylight" in out["answer"] or "night" in out["answer"]


def test_no_llm_answers_moon_question_and_declines_unknowns_honestly():
    a = Agent(None)
    assert "Moon" in a.run("what is the moon doing?")["answer"]
    assert "I can answer" in a.run("write me a poem")["answer"]


def test_groq_chat_skips_retired_models_and_remembers_the_working_one(monkeypatch):
    import io
    import urllib.error

    from terraforge.agent import core

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    tried = []

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        model = json.loads(req.data)["model"]
        tried.append(model)
        if model == "retired/model":
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"{}"))
        return Resp(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)
    chat = core.GroqChat(models=("retired/model", "good/model"))
    assert chat([], [])["content"] == "hi"
    assert tried == ["retired/model", "good/model"]
    chat([], [])
    assert tried[-1] == "good/model" and len(tried) == 3  # remembered: no retry of the dead one


def test_groq_chat_does_not_mask_auth_errors_as_missing_models(monkeypatch):
    import io
    import urllib.error

    import pytest

    from terraforge.agent import core

    monkeypatch.setenv("GROQ_API_KEY", "bad")

    def deny(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(core.urllib.request, "urlopen", deny)
    with pytest.raises(urllib.error.HTTPError):
        core.GroqChat(models=("a", "b"))([], [])


def test_bbox_area_tool():
    r = Agent(None)._run_tool("bbox_area", {"bbox": [6.0, 49.6, 6.01368, 49.609]})
    assert 95 < r["result"]["area_ha"] < 105
    assert "error" in Agent(None)._run_tool("bbox_area", {"bbox": [1, 1]})
