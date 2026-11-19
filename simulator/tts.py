"""Text-to-speech for the simulated caller: Piper only (open source, local ONNX, no API keys), with an on-disk cache.

Voice spec: "piper:<voice>", where <voice> is the model file stem in simulator/models/, e.g. piper:en_US-lessac-medium.
Get a voice with:  python -m piper.download_voices --download-dir simulator/models en_US-lessac-medium
Every clip is raw PCM16 mono 16 kHz, which is what the voice agent expects from a mic.
"""
import hashlib
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"
MODELS = Path(__file__).resolve().parent / "models"  # Piper voice files: <voice>.onnx + <voice>.onnx.json
SAMPLE_RATE = 16_000
TARGET_RMS = 3000   # speech RMS after normalization (~ -21 dBFS)
TARGET_PEAK = 16000  # peak ceiling, ~ -6 dBFS

chars_rendered = 0  # characters actually synthesized this run (cache misses only)
_voices: dict = {}  # loaded Piper voices, reused across clips


def _voice_name(spec: str) -> str:
    provider, _, name = spec.partition(":")
    if provider != "piper" or not name:
        raise ValueError(f"unsupported voice spec {spec!r}: only Piper voices are available, e.g. piper:en_US-lessac-medium")
    return name


def _cache_path(voice: str, text: str) -> Path:
    key = hashlib.sha1(f"piper:{voice}|{text}".encode()).hexdigest()[:16]
    return CACHE / f"piper_{voice}_{key}.pcm"


def _to_16k(x: np.ndarray, rate: int) -> bytes:
    """int16 or float32 [-1, 1] samples at `rate` -> raw PCM16 mono 16 kHz."""
    import soxr

    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    if rate != SAMPLE_RATE:
        x = soxr.resample(x, rate, SAMPLE_RATE)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _piper(voice: str, text: str) -> bytes:
    if voice not in _voices:
        from piper import PiperVoice

        _voices[voice] = PiperVoice.load(MODELS / f"{voice}.onnx")
    chunks = list(_voices[voice].synthesize(text))
    if not chunks:
        raise RuntimeError("piper produced no audio")
    x = np.concatenate([c.audio_int16_array for c in chunks])
    return _to_16k(x, chunks[0].sample_rate)


def available(spec: str) -> str | None:
    """Returns a reason the voice can't be used, or None if it looks usable."""
    try:
        voice = _voice_name(spec)
    except ValueError as exc:
        return str(exc)
    if not (MODELS / f"{voice}.onnx").exists():
        return f"simulator/models/{voice}.onnx missing (python -m piper.download_voices --download-dir simulator/models {voice})"
    return None


def _normalize(pcm: bytes) -> bytes:
    """Scale to a common speech level so voices of different loudness reach the agent at a similar volume."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    speech = x[np.abs(x) > 300]
    if speech.size == 0:
        return pcm
    gain = min(TARGET_RMS / np.sqrt(np.mean(speech**2)), TARGET_PEAK / max(np.abs(x).max(), 1.0))
    return np.clip(x * gain, -32768, 32767).astype("<i2").tobytes()


def synthesize(spec: str, text: str) -> bytes:
    """Render `text` with the Piper voice `spec` (level-normalized), using the cache when possible.
    The cache holds Piper's raw output; normalization is applied on the way out."""
    global chars_rendered
    voice = _voice_name(spec)
    path = _cache_path(voice, text)
    if path.exists():
        return _normalize(path.read_bytes())
    pcm = _piper(voice, text)
    chars_rendered += len(text)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pcm)
    return _normalize(pcm)
