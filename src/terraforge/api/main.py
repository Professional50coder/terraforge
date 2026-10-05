"""FastAPI service: batch inference, similarity search, RAG explanation, live stream.

Concurrency model:
- CPU-bound inference runs in a thread pool via `run_in_executor`, so the event loop
  (and every open WebSocket) stays responsive while a batch is being scored.
- Within one request, the engine splits the batch into chunks scored in parallel.
- The WebSocket stream is server-pushed: one task per connection, no polling.
"""
from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from terraforge.agent.core import Agent, GroqChat
from terraforge.data.eurosat import CLASSES
from terraforge.geo import sky
from terraforge.inference.predictor import InferenceEngine
from terraforge.inference.vector_index import VectorIndex
from terraforge.multimodal.explainer import Explainer
from terraforge.api.sources import mean_ndvi, rgb_thumbnail
from terraforge.training import drift as driftlib

MAX_BATCH = 256
DASHBOARD = Path(__file__).with_name("dashboard.html")
GLOBE = Path(__file__).with_name("globe.html")
LOGO = Path(__file__).with_name("logo.svg")


class PatchBatch(BaseModel):
    """Float32 array, little-endian, C-order, base64-encoded; raw reflectance (DN)."""
    shape: list[int] = Field(min_length=4, max_length=4)
    data_b64: str
    k: int = Field(default=5, ge=1, le=50)

    def array(self) -> np.ndarray:
        n, c, h, w = self.shape
        if not 1 <= n <= MAX_BATCH:
            raise HTTPException(422, f"batch size must be 1..{MAX_BATCH}")
        try:
            arr = np.frombuffer(base64.b64decode(self.data_b64), dtype="<f4")
            return arr.reshape(self.shape).copy()  # frombuffer views are read-only
        except Exception:
            raise HTTPException(422, "data_b64 does not match shape")


def build_reference_index(engine: InferenceEngine, source, n: int = 512) -> VectorIndex:
    """Embed `n` labelled reference patches so retrieval has something to find."""
    x, y = source.next_batch(n)
    out = engine.predict(x)
    index = VectorIndex(out["embeddings"].shape[1])
    index.add(out["embeddings"], [{"class": CLASSES[int(i)], "ref_id": j}
                                  for j, i in enumerate(y)])
    return index


class AgentIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[dict] = Field(default_factory=list, max_length=12)


def default_agent(analyzer=None) -> Agent:
    """LLM-backed when GROQ_API_KEY is present in the environment, rule-based otherwise."""
    from terraforge.agent.tools import build_tools
    return Agent(GroqChat() if os.environ.get("GROQ_API_KEY") else None,
                 build_tools(analyzer=analyzer))


def create_app(engine: InferenceEngine, source, index: VectorIndex | None = None,
               explainer: Explainer | None = None, drift_reference=None,
               agent: Agent | None = None, stac=None) -> FastAPI:
    from terraforge.data.stac_client import STACClient
    stac = stac or STACClient()
    app = FastAPI(title="TerraForge", version="0.1.0")
    explainer = explainer or Explainer("template")
    untrained = "untrained" in engine.version

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD.read_text(encoding="utf-8")

    @app.get("/logo.svg")
    def logo():
        from fastapi.responses import Response
        return Response(LOGO.read_bytes(), media_type="image/svg+xml")

    @app.get("/globe", response_class=HTMLResponse)
    def globe_page():
        return GLOBE.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": engine.version, "untrained": untrained,
                "source": source.kind, "index_size": len(index) if index else 0,
                "explainer": explainer.backend}

    @app.get("/sky")
    def sky_state(time: str | None = None):
        """Sun subpoint, day/night terminator and moon phase (computed, not simulated)."""
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(time.replace("Z", "+00:00")) if time else None
        except ValueError:
            raise HTTPException(422, "time must be ISO-8601, e.g. 2024-06-21T12:00:00Z")
        return {"sun": sky.sun_position(dt), "terminator": sky.terminator(dt, 120),
                "moon": sky.moon_phase(dt)}

    def analyze_sync(lat: float, lon: float) -> dict:
        """Real chip -> classify -> explain, with caveats. Raises LookupError (no scene),
        OSError/KeyError (upstream imagery problem). Shared by /analyze and the agent."""
        from datetime import date, timedelta

        from terraforge.data.chip_fetcher import fetch_chip
        from terraforge.data.stac_client import SceneQuery
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("lat must be in [-90,90] and lon in [-180,180]")
        q = SceneQuery((lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002),
                       str(date.today() - timedelta(days=90)), str(date.today()),
                       max_cloud=40, limit=10)
        items = stac.search(q)
        if not items:
            raise LookupError("no recent low-cloud Sentinel-2 scene covers this point")
        item = items[0]
        hrefs = {k: a.href for k, a in item.assets.items()}
        baseline = item.properties.get("s2:processing_baseline")
        chip, meta = fetch_chip(hrefs, lon, lat, float(baseline) if baseline else None)
        out = engine.predict(chip[None])
        ndvi = mean_ndvi(chip)
        neigh = index.search(out["embeddings"], 5)[0] if index and len(index) else []
        expl = explainer.explain(out["labels"][0], float(out["confidence"][0]), ndvi, neigh)
        warnings = ["Trained on Level-1C data; this chip is Level-2A, so accuracy is lower."]
        if meta["missing_bands"]:
            warnings.append("Missing band(s) filled with zeros: " + ", ".join(meta["missing_bands"]))
        if meta["cloud_fraction"] > 0.3:
            warnings.append(f"{meta['cloud_fraction']:.0%} of this area is cloudy; treat as unreliable.")
        if untrained:
            warnings.append("The model is not trained yet; this prediction is not meaningful.")
        pset = out["sets"][0] if out["sets"] else None
        if pset:
            a = engine.conformal["alpha"]
            warnings.append(f"The answer set covers the true class at least {1 - a:.0%} of the time "
                            "on data like the calibration set; that guarantee does not hold under "
                            "haze, missing bands or the Level-2A shift.")
        return {"prediction_set": pset, "scene": {"id": item.id, "date": str(item.datetime)[:10]},
                "label": out["labels"][0], "confidence": float(out["confidence"][0]),
                "ndvi": ndvi, "cloud_fraction": meta["cloud_fraction"],
                "explanation": expl["text"], "explanation_source": expl["source"],
                "warnings": warnings, "untrained": untrained}

    agent = agent or default_agent(analyze_sync)

    @app.get("/analyze")
    async def analyze(lat: float, lon: float):
        """Fetch a real Sentinel-2 chip at a point, classify it, explain it, with caveats."""
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(422, "lat must be in [-90,90] and lon in [-180,180]")
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, analyze_sync, lat, lon)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except (OSError, KeyError) as e:  # network / missing asset: upstream problem, not ours
            raise HTTPException(502, f"could not fetch imagery: {type(e).__name__}")

    @app.post("/agent")
    async def agent_endpoint(body: AgentIn):
        """Conversational entry point (typed or transcribed speech). Blocking LLM/tool work
        runs in a thread so the event loop and WebSocket streams stay responsive."""
        loop = asyncio.get_running_loop()
        safe_history = [{"role": h["role"], "content": str(h.get("content", ""))[:500]}
                        for h in body.history if h.get("role") in ("user", "assistant")]
        return await loop.run_in_executor(None, agent.run, body.message, safe_history)

    @app.post("/predict")
    async def predict(batch: PatchBatch):
        x = batch.array()
        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(None, engine.predict, x)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {
            "model_version": out["model_version"], "latency_ms": out["latency_ms"],
            "untrained": untrained,
            "conformal_alpha": engine.conformal["alpha"] if engine.conformal else None,
            "predictions": [{"label": l, "confidence": float(c),
                             "prediction_set": out["sets"][i] if out["sets"] else None,
                             "probs": dict(zip(CLASSES, map(float, p)))}
                            for i, (l, c, p) in enumerate(zip(out["labels"], out["confidence"],
                                                              out["probs"]))],
        }

    @app.post("/similar")
    async def similar(batch: PatchBatch):
        if not index or not len(index):
            raise HTTPException(503, "vector index is empty")
        x = batch.array()
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, engine.predict, x)
        return {"neighbours": index.search(out["embeddings"], batch.k)}

    @app.post("/explain")
    async def explain(batch: PatchBatch):
        x = batch.array()
        if len(x) != 1:
            raise HTTPException(422, "explain takes exactly one patch")
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, engine.predict, x)
        neigh = index.search(out["embeddings"], batch.k)[0] if index and len(index) else []
        result = await loop.run_in_executor(
            None, explainer.explain, out["labels"][0], float(out["confidence"][0]),
            mean_ndvi(x[0]), neigh)
        return {**result, "untrained": untrained}

    class FactsIn(BaseModel):
        label: str
        confidence: float = Field(ge=0, le=1)
        ndvi: float | None = None
        neighbours: list[dict] = []

    @app.post("/explain_facts")
    async def explain_facts(f: FactsIn):
        """Explain an already-scored patch from its computed facts (used by the dashboard)."""
        if f.label not in CLASSES:
            raise HTTPException(422, "unknown class")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, explainer.explain, f.label, f.confidence, f.ndvi, f.neighbours)
        return {**result, "untrained": untrained}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket, batch: int = 4, interval: float = 0.8):
        await ws.accept()
        batch, interval = max(1, min(batch, 32)), max(0.1, interval)
        loop = asyncio.get_running_loop()
        counts = {c: 0 for c in CLASSES}
        seen = correct = labelled = 0
        window: list[np.ndarray] = []
        psi_max = None
        try:
            await ws.send_json({"type": "hello", "model_version": engine.version,
                                "untrained": untrained, "source": source.kind,
                                "classes": list(CLASSES)})
            while True:
                x, y = await loop.run_in_executor(None, source.next_batch, batch)
                out = await loop.run_in_executor(None, engine.predict, x)
                items = []
                neigh = index.search(out["embeddings"], 5) if index and len(index) \
                    else [[] for _ in range(len(x))]
                for i in range(len(x)):
                    counts[out["labels"][i]] += 1
                    seen += 1
                    truth = None if y is None else CLASSES[int(y[i])]
                    if truth is not None:
                        labelled += 1
                        correct += int(truth == out["labels"][i])
                    items.append({
                        "rgb_b64": base64.b64encode(rgb_thumbnail(x[i])).decode(),
                        "size": x.shape[-1], "label": out["labels"][i],
                        "confidence": float(out["confidence"][i]), "truth": truth,
                        "ndvi": mean_ndvi(x[i]),
                        "neighbours": [{"class": n["class"], "score": n["score"]}
                                       for n in neigh[i]]})
                if drift_reference is not None:
                    window.append(x.mean(axis=(2, 3)))
                    window = window[-25:]
                    if len(window) >= 5:
                        live = np.concatenate(window)
                        psi_max = float(driftlib.psi(live, *drift_reference).max())
                await ws.send_json({
                    "type": "batch", "items": items, "latency_ms": out["latency_ms"],
                    "counts": counts, "seen": seen,
                    "running_accuracy": (correct / labelled) if labelled else None,
                    "drift_psi": psi_max,
                    "drift_status": None if psi_max is None else driftlib.status(psi_max)})
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
