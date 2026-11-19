"""Caller clip library: fixed, indexed recordings of what the caller says, replayed into the agent for evaluation.

The caller's words are the test INPUT; the agent's reply is what is under test. Storing the input as a checksummed WAV
makes every run replay exactly the same audio (no TTS drift, no API calls, no keys) and lets real human recordings
sit beside synthetic ones.

    python -m simulator.clips record [--voices kokoro:af_heart,piper:en_US-lessac-medium] [--scenarios id,id]
        render the scenarios' caller turns with TTS voices and store them (level-normalised, 16 kHz mono)
    python -m simulator.clips add-human dishwasher_request evals/my_recording.wav --name alice [--turn 0]
        import a real recording (any WAV/FLAC, any rate) for a scenario line; it must say the scripted sentence.
        --turn N is the line number: the turn of a fixed scenario, or the position in a reactive dialogue's list of
        lines (an unknown number prints that list)
    python -m simulator.clips list | verify | export <out.html>

Replay in an eval with a clip voice:   run_evals.py --voices clip:kokoro:af_heart,clip:human:alice   (or clip:*)
A clip is found by (source, text): source is "kokoro:af_heart" for TTS or "human:<name>" for a person.
Files live in evals/caller_clips/<id>.wav with index.json beside them; track *.wav with Git LFS.
"""
import argparse
import hashlib
import json
import re
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from .dialogue import Script

HERE = Path(__file__).resolve().parent
DIR = HERE / "caller_clips"
INDEX = DIR / "index.json"
RATE = 16_000
DEFAULT_VOICES = "kokoro:af_heart,kokoro:am_michael,piper:en_US-lessac-medium,piper:en_GB-alba-medium"


def tkey(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")


def load_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {"clips": {}}


def _save_index(idx: dict) -> None:
    DIR.mkdir(exist_ok=True)
    INDEX.write_text(json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write(clip_id: str, pcm: bytes) -> Path:
    DIR.mkdir(exist_ok=True)
    p = DIR / f"{_slug(clip_id)}.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return p


def _read(entry: dict) -> bytes:
    with wave.open(str(DIR / entry["file"])) as w:
        return w.readframes(w.getnframes())


def find(source: str, text: str) -> dict | None:
    """The index entry for this source saying this text, or None."""
    want = tkey(text)
    return next((c for c in load_index()["clips"].values() if c["source"] == source and tkey(c["text"]) == want), None)


def get(source: str, text: str) -> bytes:
    """Raw PCM16 16 kHz for a stored clip, checksum-verified. Raises if missing or altered."""
    entry = find(source, text)
    if entry is None:
        raise LookupError(f"no stored clip for source {source!r} saying {text[:50]!r}; run: python -m simulator.clips record")
    pcm = _read(entry)
    if _sha(pcm) != entry["sha256"]:
        raise ValueError(f"clip {entry['id']} no longer matches its checksum (file changed); re-record or fix index.json")
    return pcm


def sources() -> list[str]:
    return sorted({c["source"] for c in load_index()["clips"].values()})


def expand(spec: str) -> list[str]:
    """`clip:*` / `clip:human:*` -> concrete clip:<source> voice specs; anything else passes through."""
    if spec == "clip:*":
        return [f"clip:{s}" for s in sources()]
    if spec.startswith("clip:") and spec.endswith(":*"):
        pre = spec[5:-1]
        return [f"clip:{s}" for s in sources() if s.startswith(pre)]
    return [spec]


def _put(scenario: str, turn: int, source: str, text: str, pcm: bytes, reactive: bool = False) -> dict:
    clip_id = f"{scenario}.{'r' if reactive else 't'}{turn}.{source}"  # r<n> = line n of a reactive dialogue
    p = _write(clip_id, pcm)
    entry = {"id": clip_id, "scenario": scenario, "turn": turn, "text": text, "source": source, "file": p.name,
             "sha256": _sha(pcm), "duration_s": round(len(pcm) / 2 / RATE, 2), "rate": RATE,
             "created": datetime.now().isoformat(timespec="seconds")}
    idx = load_index()
    idx["clips"][clip_id] = entry
    _save_index(idx)
    return entry


def _scenarios() -> list[dict]:
    return json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ commands ---
def record(voices: list[str], scenario_ids: list[str] | None) -> None:
    from . import tts

    n = 0
    for v in voices:
        why = tts.available(v)
        if why:
            print(f"skipping {v}: {why}")
            continue
        for s in _scenarios():
            if scenario_ids and s["id"] not in scenario_ids:
                continue
            for i, text in enumerate(Script(s).lines()):
                e = _put(s["id"], i, v, text, tts.synthesize(v, text), "dialogue" in s)
                n += 1
                print(f"  {e['id']:<60} {e['duration_s']:>5.1f}s")
    print(f"recorded {n} clip(s) -> {DIR}")


def add_human(scenario: str, wav: Path, name: str, turn: int = 0) -> None:
    import soundfile as sf
    import soxr

    from . import tts

    sc = next((s for s in _scenarios() if s["id"] == scenario), None)
    lines = Script(sc).lines() if sc else []
    if sc is None or turn >= len(lines):
        listed = "".join(f"\n  {i}: {t}" for i, t in enumerate(lines))
        raise SystemExit(f"unknown scenario/line: {scenario!r} #{turn}; scenarios: {', '.join(s['id'] for s in _scenarios())}"
                         + (f"\nlines of {scenario}:{listed}" if sc else ""))
    x, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                            # to mono
    if rate != RATE:
        x = soxr.resample(x, rate, RATE)
    pcm = tts._normalize((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    e = _put(scenario, turn, f"human:{name}", lines[turn], pcm, "dialogue" in sc)
    print(f"added {e['id']} ({e['duration_s']}s). It must say exactly: {e['text']!r}")


def listing() -> None:
    clips = sorted(load_index()["clips"].values(), key=lambda c: (c["scenario"], c["turn"], c["source"]))
    for c in clips:
        print(f"{c['scenario']:<22} t{c['turn']}  {c['source']:<28} {c['duration_s']:>5.1f}s  {c['sha256'][:10]}  {c['text'][:50]}")
    print(f"{len(clips)} clip(s) from {len(sources())} source(s): {', '.join(sources()) or '-'}")


def verify() -> int:
    bad = 0
    for c in load_index()["clips"].values():
        f = DIR / c["file"]
        if not f.exists():
            print(f"MISSING  {c['id']}")
            bad += 1
        elif _sha(_read(c)) != c["sha256"]:
            print(f"CHANGED  {c['id']}")
            bad += 1
    print("all clips intact" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def export_html(out: Path) -> None:
    import html

    clips = sorted(load_index()["clips"].values(), key=lambda c: (c["scenario"], c["turn"], c["source"]))
    rel = Path(DIR).relative_to(out.resolve().parent) if out.resolve().parent in DIR.parents else DIR
    rows, last = [], None
    for c in clips:
        if (c["scenario"], c["turn"]) != last:
            last = (c["scenario"], c["turn"])
            rows.append(f'<h3 style="margin-bottom:4px">{html.escape(c["scenario"])} · turn {c["turn"]} '
                        f'<small style="font-weight:400;color:#666">“{html.escape(c["text"])}”</small></h3>')
        rows.append(f'<div style="display:flex;gap:12px;align-items:center"><span style="width:220px"><b>{html.escape(c["source"])}</b>'
                    f'<br><small style="color:#666">{c["duration_s"]}s</small></span>'
                    f'<audio controls preload="none" src="{rel.as_posix()}/{c["file"]}" style="flex:1"></audio></div>')
    out.write_text('<!doctype html><meta charset=utf-8><title>Caller clips</title><body style="font:15px system-ui;max-width:820px;'
                   'margin:24px auto;padding:0 16px"><h2>Caller clip library</h2>' + "".join(rows) + "</body>", encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--voices", default=DEFAULT_VOICES)
    r.add_argument("--scenarios", default="all")
    h = sub.add_parser("add-human")
    h.add_argument("scenario")
    h.add_argument("wav", type=Path)
    h.add_argument("--name", required=True)
    h.add_argument("--turn", type=int, default=0)
    sub.add_parser("list")
    sub.add_parser("verify")
    e = sub.add_parser("export")
    e.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.voices.split(","), None if a.scenarios == "all" else a.scenarios.split(","))
    elif a.cmd == "add-human":
        add_human(a.scenario, a.wav, a.name, a.turn)
    elif a.cmd == "list":
        listing()
    elif a.cmd == "verify":
        return verify()
    else:
        export_html(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
