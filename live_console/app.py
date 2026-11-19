"""Live web view of a simulated call: simulator (persona) | agent under test | LLM judge.

    python -m live_console.app            # then open http://127.0.0.1:8100
    python -m live_console.app --port 8100 --agent-url ws://127.0.0.1:8000/ws

The page starts a run (POST /api/run) and follows it over Server-Sent Events (GET /api/stream/<run_id>). Events are
replayed from the start, so a late or reloaded page still shows the whole call. The voice agent must be running
(cd voice_agent; python -m uvicorn main:app --port 8000). The finished run is saved to results/<timestamp>/<persona>.json.
"""
import argparse
import asyncio
import contextlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from judge import llm_judge as judge  # noqa: E402
from simulator import sim_runner  # noqa: E402

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Voice simulator console")
RUNS: dict[str, dict] = {}  # run_id -> {"events": [...], "done": bool}
CFG = {"agent_url": "ws://127.0.0.1:8000/ws", "persona_url": "http://127.0.0.1:8080/persona-caller/turn"}


class RunRequest(BaseModel):
    persona: str
    audio: str = "hybrid"                       # static | dynamic | hybrid
    voice: str = "piper:en_US-lessac-medium"
    persona_backend: str = "ollama:qwen2:7b"    # "ollama:<model>" or "agent" (base-agent persona-caller over HTTP)
    judge_model: str = judge.MODEL
    max_turns: int = 6


def _personas() -> list[dict]:
    return json.loads((HERE.parent / "simulator" / "personas.json").read_text(encoding="utf-8"))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/api/personas")
async def personas() -> list[dict]:
    return [{"id": p["id"], "name": p["persona"]["name"], "goal": p["scenario"]["goal"]} for p in _personas()]


async def _execute(run_id: str, req: RunRequest) -> None:
    run = RUNS[run_id]

    def emit(ev: dict) -> None:
        run["events"].append(ev)

    sp = next(p for p in _personas() if p["id"] == req.persona)
    args = argparse.Namespace(
        url=CFG["agent_url"], voice=req.voice, audio=req.audio, max_turns=req.max_turns,
        persona_url=CFG["persona_url"], persona_ollama=req.persona_backend if req.persona_backend.startswith("ollama:") else "",
    )
    out = HERE.parent / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    run["wav"] = out / f"{sp['id']}.wav"

    def on_audio(channel: int, pcm: bytes) -> None:  # live audio for the browser: 1 byte channel + raw PCM16
        for q in run["subs"]:
            q.put_nowait(bytes([channel]) + pcm)

    try:
        result = await sim_runner.simulate(sp, args, emit, on_audio=on_audio, wav_path=run["wav"])
        if run["wav"].exists():
            emit({"type": "recording", "url": f"/api/recording/{run_id}"})
        if result.get("conversation"):
            emit({"type": "judge_start", "model": req.judge_model})
            verdict = await asyncio.to_thread(
                judge.evaluate, {"id": sp["id"], "description": sp["scenario"]["goal"]}, result, req.judge_model)
            result["judge"] = verdict
            emit({"type": "judge_result", **verdict})
        (out / f"{sp['id']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit({"type": "saved", "path": str(out / f"{sp['id']}.json")})
    except Exception as exc:  # surface any failure in the page instead of a silent hang
        emit({"type": "fatal", "message": f"{type(exc).__name__}: {str(exc)[:300]}"})
    finally:
        run["done"] = True
        for q in run["subs"]:
            q.put_nowait(None)  # tells audio listeners the call is over


@app.websocket("/api/audio/{run_id}")
async def audio(ws: WebSocket, run_id: str) -> None:
    """Live call audio for the page: binary frames, byte 0 = channel (0 caller 16 kHz, 1 agent 24 kHz), then PCM16."""
    run = RUNS.get(run_id)
    if run is None:
        await ws.close(code=1008)
        return
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    run["subs"].append(q)
    try:
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            await ws.send_bytes(chunk)
    except WebSocketDisconnect:
        pass
    finally:
        if q in run["subs"]:
            run["subs"].remove(q)
        with contextlib.suppress(Exception):
            await ws.close()


@app.get("/api/recording/{run_id}")
async def recording(run_id: str) -> FileResponse:
    run = RUNS.get(run_id)
    if run is None or not run.get("wav") or not run["wav"].exists():
        raise HTTPException(404, "no recording")
    return FileResponse(run["wav"], media_type="audio/wav")


@app.post("/api/run")
async def start_run(req: RunRequest) -> dict:
    if not any(p["id"] == req.persona for p in _personas()):
        raise HTTPException(404, "unknown persona")
    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {"events": [], "done": False, "subs": []}
    asyncio.create_task(_execute(run_id, req))
    return {"run_id": run_id}


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    run = RUNS[run_id]

    async def gen():
        sent = 0
        while True:
            while sent < len(run["events"]):
                yield f"data: {json.dumps(run['events'][sent])}\n\n"
                sent += 1
            if run["done"] and sent >= len(run["events"]):
                yield 'data: {"type": "end"}\n\n'
                return
            await asyncio.sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--agent-url", default=CFG["agent_url"])
    ap.add_argument("--persona-url", default=CFG["persona_url"])
    a = ap.parse_args()
    CFG.update(agent_url=a.agent_url, persona_url=a.persona_url)
    uvicorn.run(app, host="127.0.0.1", port=a.port)
