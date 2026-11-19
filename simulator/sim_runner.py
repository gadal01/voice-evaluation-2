"""Persona-driven caller simulator: an LLM plays the caller, Piper (or any tts.py voice) speaks it, the live agent answers.

    python -m simulator.sim_runner                                   # all personas in personas.json, default voice
    python -m simulator.sim_runner --persona sam_ac_impatient --voice piper:en_US-lessac-medium --audio hybrid
    python -m simulator.sim_runner --persona-url http://host/persona-caller/turn      # call the base-agent persona-caller
    python -m simulator.sim_runner --persona-ollama ollama:qwen2:7b   # no base-agent: same prompts, local Ollama model

Per turn: decide the next caller line (persona-caller agent or local Ollama) -> pick audio (stored clip or live TTS
render) -> stream it to the agent over the WebSocket (ws_client.py) -> wait for the reply -> repeat until the persona
ends the call or --max-turns. Then the finished call is scored by the local judge (judge.py, --judge-model).

Audio modes: static = stored clips only (clips.py), dynamic = always render with TTS, hybrid = clip if the persona picked
one that says exactly the same words, otherwise render. Start the voice agent first: cd voice_agent; python -m uvicorn main:app --port 8000
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import websockets

from judge import llm_judge as judge  # noqa: E402
from ws_client import Call, TURN_TIMEOUT_S, get_token  # noqa: E402

from . import clips, tts  # noqa: E402

HERE = Path(__file__).resolve().parent
PERSONA_DIR = HERE / "persona_agent"
FALLBACK_LINE = "Sorry, could you say that again?"
INTENTS = {"greeting", "provide_info", "confirm", "correct", "ask_question", "express_frustration", "clarify",
           "request_human", "end_call"}


# ------------------------------------------------------------ persona turn ---
def build_request(persona: dict, scenario: dict, transcript: list[dict], catalog: list[dict], turn: int, max_turns: int) -> dict:
    """The persona-caller request. Structured values are pre-formatted to text so any prompt template can embed them."""
    note = "This is the final allowed turn: say goodbye and end the call." if turn + 1 >= max_turns else ""
    return {"user": {"userId": "sim", "userEmail": os.getenv("SIM_USER_EMAIL", "sim@example.com")}, "data": {
        "persona": {**persona, "facts": "; ".join(f"{k}: {v}" for k, v in persona.get("facts", {}).items())
                    if isinstance(persona.get("facts"), dict) else persona.get("facts", "")},
        "scenario": scenario,
        "clip_catalog": "\n".join(f'- {c["clip_id"]}: "{c["text"]}"' for c in catalog) or "(none)",
        "transcript": "\n".join(f'{t["speaker"]}: {t["text"]}' for t in transcript) or "(the call has just connected; you speak first)",
        "turn_index": turn, "max_turns": max_turns, "turn_note": note}}


def normalize_turn(raw: dict) -> dict:
    """Coerce a model reply to the turn_output.json shape; raises ValueError if there is nothing sayable."""
    text = str(raw.get("utterance", "")).strip()
    if not text:
        raise ValueError("empty utterance")
    intent = raw.get("intent") if raw.get("intent") in INTENTS else "provide_info"
    return {"utterance": text[:400], "clip_id": raw.get("clip_id") or None, "intent": intent,
            "end_call": bool(raw.get("end_call")), "barge_in": bool(raw.get("barge_in")),
            "hesitation_ms": max(0, min(3000, int(raw.get("hesitation_ms") or 0))),
            "behavior_tags": [str(t) for t in raw.get("behavior_tags", [])][:6]}


def _fill(template: str, data: dict) -> str:
    def get(m):
        v = data
        for part in m.group(1).split("."):
            v = v.get(part, "") if isinstance(v, dict) else ""
        return str(v)
    return re.sub(r"\{\{\s*data\.([\w.]+)\s*\}\}", get, template)


def turn_via_ollama(model: str, req: dict) -> dict:
    """Same prompts as the base-agent persona-caller, rendered locally and run on an Ollama model."""
    d = PERSONA_DIR / "prompts" / "plan_turn"
    system = (d / "system.txt").read_text(encoding="utf-8")
    user = _fill((d / "user.txt").read_text(encoding="utf-8"), req["data"])
    text = judge._ollama(model[len("ollama:"):], system, user)
    return normalize_turn(json.loads(re.search(r"\{.*\}", text, re.S).group(0)))


def turn_via_agent(url: str, req: dict) -> dict:
    r = httpx.post(url, json=req, timeout=60, headers={"Authorization": f"Bearer {os.getenv('PERSONA_TOKEN', '')}"}
                   if os.getenv("PERSONA_TOKEN") else {})
    r.raise_for_status()
    body = r.json()
    out = body.get("result", body) if isinstance(body, dict) else body
    if isinstance(out, str):
        out = json.loads(out)
    return normalize_turn(out)


# -------------------------------------------------------------------- audio ---
def clip_catalog(source: str) -> list[dict]:
    """Stored clips for this voice that the persona may reuse, as [{clip_id, text}] (deduplicated by wording)."""
    seen: dict[str, dict] = {}
    for c in clips.load_index()["clips"].values():
        if c["source"] == source:
            seen.setdefault(clips.tkey(c["text"]), {"clip_id": c["id"], "text": c["text"]})
    return list(seen.values())


def pick_audio(turn: dict, voice: str, mode: str, catalog: list[dict]) -> tuple[bytes, str]:
    """Returns (PCM16 16 kHz, how) where how is 'clip' or 'tts'. A clip is used only if its words match the utterance."""
    entry = next((c for c in catalog if c["clip_id"] == turn["clip_id"]), None) if turn["clip_id"] else None
    if mode != "dynamic" and entry and clips.tkey(entry["text"]) == clips.tkey(turn["utterance"]):
        return clips.get(voice, entry["text"]), "clip"
    if mode == "static":
        found = clips.find(voice, turn["utterance"])  # the model reworded a stored line: still use it if it exists
        if found:
            return clips.get(voice, found["text"]), "clip"
        raise LookupError(f"static mode: no stored clip for {turn['utterance'][:60]!r}")
    return tts.synthesize(voice, turn["utterance"]), "tts"


# --------------------------------------------------------------- one call ---
def transcript_of(call: Call, said: list[str]) -> list[dict]:
    out = []
    for i, line in enumerate(said):
        out.append({"speaker": "caller", "text": line})
        reply = " ".join(e["text"] for e in call.events if e["turn"] == i and e["type"] == "reply")
        if reply:
            out.append({"speaker": "agent", "text": reply})
    return out


def conversation_of(call: Call, said: list[str]) -> list[dict]:
    """The shape judge.py renders: caller / heard / agent / tool items in time order."""
    conv = []
    for i, line in enumerate(said):
        conv.append({"role": "caller", "turn": i, "t": call.user_start.get(i, 0), "text": line})
    for e in call.events:
        if e["type"] == "transcript":
            conv.append({"role": "heard", "t": e["t"], "text": e["text"]})
        elif e["type"] == "reply":
            conv.append({"role": "agent", "t": e["t"], "text": e["text"]})
        elif e["type"] == "tool":
            conv.append({"role": "tool", "t": e["t"], "name": e["name"], "args": e["args"], "result": e["result"]})
    return sorted(conv, key=lambda x: x["t"])


async def simulate(sp: dict, args: argparse.Namespace, emit=lambda ev: None, on_audio=None, wav_path=None) -> dict:
    """Runs one simulated call. `emit(event_dict)` is called live for the web console (events: call_start,
    persona_thinking, persona_turn, caller_speaking, caller_spoken, agent_event, agent_audio, turn_end, call_end)."""
    persona, scenario = sp["persona"], sp["scenario"]
    voice = sp.get("voice") or args.voice
    source = voice[len("clip:"):] if voice.startswith("clip:") else voice
    catalog = clip_catalog(source) if args.audio != "dynamic" else []
    ask = (lambda req: turn_via_ollama(args.persona_ollama, req)) if args.persona_ollama else \
          (lambda req: turn_via_agent(args.persona_url, req))
    call, said, trace = Call(), [], []
    call.on_audio = on_audio  # live audio tap for the web console: (channel 0 caller 16 kHz | 1 agent 24 kHz, pcm)
    seen, audio_noted = 0, set()

    def pump(turn: int) -> None:  # forward new agent-side events to the console as they arrive
        nonlocal seen
        while seen < len(call.events):
            e = call.events[seen]
            seen += 1
            emit({"type": "agent_event", "kind": e["type"], "turn": e["turn"], "t": round(e["t"], 2),
                  **{k: e[k] for k in ("text", "name", "args", "result", "ms", "message") if k in e}})
        if turn in call.first_audio and turn not in audio_noted:
            audio_noted.add(turn)
            ok = turn in call.user_end and call.first_audio[turn] >= call.user_end[turn]
            emit({"type": "agent_audio", "turn": turn,
                  "ttfa_ms": round((call.first_audio[turn] - call.user_end[turn]) * 1000) if ok else None})

    emit({"type": "call_start", "id": sp["id"], "persona": persona, "scenario": scenario, "voice": voice,
          "audio": args.audio, "max_turns": args.max_turns})
    async with websockets.connect(args.url, max_size=None,
                                  additional_headers={"Authorization": f"Bearer {await get_token(args.url)}"}) as ws:
        rx = asyncio.create_task(call.receive(ws))
        tx = None
        try:
            await asyncio.wait_for(call.ready.wait(), 30)
            if call.error:
                emit({"type": "call_end", "error": call.error})
                return {"id": sp["id"], "error": call.error}
            tx = asyncio.create_task(call.send_loop(ws))
            for i in range(args.max_turns):
                req = build_request(persona, scenario, transcript_of(call, said), catalog, i, args.max_turns)
                emit({"type": "persona_thinking", "turn": i})
                started = time.perf_counter()
                try:
                    turn = await asyncio.to_thread(ask, req)
                except Exception as exc:  # persona service down or unparseable: keep the call going with a safe line
                    print(f"  persona turn failed ({type(exc).__name__}: {str(exc)[:100]}), using fallback line")
                    emit({"type": "persona_error", "turn": i, "message": f"{type(exc).__name__}: {str(exc)[:160]}"})
                    turn = normalize_turn({"utterance": FALLBACK_LINE, "intent": "clarify"})
                think_ms = round((time.perf_counter() - started) * 1000)
                try:
                    pcm, how = await asyncio.to_thread(pick_audio, turn, source, args.audio, catalog)
                except Exception as exc:
                    call.events.append({"type": "sim_error", "turn": i, "t": call.now(), "message": str(exc)})
                    emit({"type": "persona_error", "turn": i, "message": str(exc)})
                    break
                fields = {k: turn[k] for k in ("utterance", "clip_id", "intent", "end_call", "barge_in",
                                               "hesitation_ms", "behavior_tags")}
                emit({"type": "persona_turn", "turn": i, "how": how, "think_ms": think_ms, **fields})
                await asyncio.sleep(0.5 + turn["hesitation_ms"] / 1000)
                if turn["barge_in"] and i > 0:  # impatient caller: cut in shortly after the agent starts talking
                    for _ in range(40):
                        pump(i - 1)
                        if i - 1 in call.first_audio and call.now() > call.first_audio[i - 1] + 0.8:
                            break
                        await asyncio.sleep(0.1)
                call.turn = i
                said.append(turn["utterance"])
                call.user_start[i] = call.now()
                emit({"type": "caller_speaking", "turn": i, "seconds": round(len(pcm) / 32000, 1)})
                speak = asyncio.create_task(call.say(pcm))
                while not speak.done():
                    pump(i)
                    await asyncio.sleep(0.1)
                call.user_end[i] = speak.result()
                emit({"type": "caller_spoken", "turn": i})
                trace.append({"turn": i, "how": how, "think_ms": think_ms, **fields})
                print(f"  [{i}] ({how}) {turn['utterance']}")
                deadline = time.perf_counter() + TURN_TIMEOUT_S
                while not call.turn_done(i) and not call.error and time.perf_counter() < deadline and not rx.done():
                    pump(i)
                    await asyncio.sleep(0.1)
                pump(i)
                for e in call.events:  # show what the agent did this turn
                    if e["turn"] == i and e["type"] == "tool":
                        print(f"      tool {e['name']}({json.dumps(e['args'])}) -> {json.dumps(e['result'])[:120]}  [{e.get('ms')} ms]")
                heard = " ".join(e["text"] for e in call.events if e["turn"] == i and e["type"] == "transcript")
                reply = " ".join(e["text"] for e in call.events if e["turn"] == i and e["type"] == "reply")
                ttfa = call.first_audio.get(i)
                print(f"      agent heard: {heard}")
                print(f"      AGENT: {reply}")
                if ttfa and i in call.user_end:
                    print(f"      (first audio {round((ttfa - call.user_end[i]) * 1000)} ms after the caller stopped)")
                if not call.turn_done(i):
                    call.events.append({"type": "timeout", "turn": i, "t": call.now()})
                    emit({"type": "agent_event", "kind": "timeout", "turn": i, "message": "agent did not finish in time"})
                    break
                emit({"type": "turn_end", "turn": i})
                if turn["end_call"]:
                    break
        finally:
            if tx:
                tx.cancel()
            rx.cancel()
    if wav_path:
        from ws_client import write_conversation_wav
        write_conversation_wav(call, Path(wav_path))
    conv = conversation_of(call, said)
    result = {"id": sp["id"], "voice": voice, "audio_mode": args.audio, "turns": trace, "conversation": conv,
              "error": call.error, "timed_out": any(e["type"] == "timeout" for e in call.events),
              "tools_called": [e["name"] for e in call.events if e["type"] == "tool"]}
    emit({"type": "call_end", "tools_called": result["tools_called"], "timed_out": result["timed_out"], "error": call.error})
    return result


# --------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="voice agent WebSocket")
    ap.add_argument("--persona", default="", help="comma-separated persona ids from personas.json (default: all)")
    ap.add_argument("--voice", default="piper:en_US-lessac-medium", help="tts.py voice spec, or clip:<source> for stored clips")
    ap.add_argument("--audio", choices=["static", "dynamic", "hybrid"], default="hybrid")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--persona-url", default=os.getenv("PERSONA_URL", "http://127.0.0.1:8080/persona-caller/turn"))
    ap.add_argument("--persona-ollama", default="", help="e.g. ollama:qwen2:7b: skip base-agent, run the persona prompts locally")
    ap.add_argument("--judge-model", default="ollama:qwen2:7b")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    args = ap.parse_args()

    personas = json.loads((HERE / "personas.json").read_text(encoding="utf-8"))
    want = {p for p in args.persona.split(",") if p}
    personas = [p for p in personas if not want or p["id"] in want]
    if not personas:
        sys.exit("no matching persona in personas.json")
    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    for sp in personas:
        print(f"== {sp['id']} (audio={args.audio})")
        run = asyncio.run(simulate(sp, args))
        if not args.no_judge and run.get("conversation"):
            run["judge"] = judge.evaluate({"id": sp["id"], "description": sp["scenario"]["goal"]}, run, args.judge_model)
            j = run["judge"]
            print("  judge:", f"{j['overall']:.1f} {'PASS' if j['pass'] else 'FAIL'} {j['issues']}" if "scores" in j else j["error"])
        (out / f"{sp['id']}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
