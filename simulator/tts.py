"""Text-to-speech for the simulated caller: Amazon Polly and ElevenLabs, with an on-disk cache.

Voice spec strings:
  polly:Joanna            neural engine (default)
  polly:Ruth:generative   explicit engine
  eleven:Rachel           a named premade voice below, or a raw ElevenLabs voice id
  kokoro:af_heart         Kokoro (open source, local ONNX); any voice id from voices-v1.0.bin, e.g. af_heart, am_michael, bf_emma
  piper:en_US-lessac-medium   Piper (open source, local ONNX); the voice name is the model file stem in evals/models/
  clip:kokoro:af_heart    replay a stored caller recording (evals/caller_clips, see clips.py); clip:human:alice, clip:*
  sapi:Zira               built-in Windows voice (name substring); free and offline, for harness dev only
Local models live in evals/models/ (git-ignore it): kokoro-v1.0.onnx and voices-v1.0.bin from the kokoro-onnx GitHub
release "model-files-v1.0", and Piper voices via  python -m piper.download_voices --download-dir evals/models en_US-lessac-medium
Every clip is raw PCM16 mono 16 kHz, which is what the voice agent expects from a mic.
"""
import hashlib
import os
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CACHE = Path(__file__).resolve().parent / "cache"
MODELS = Path(__file__).resolve().parent / "models"  # local TTS model files (see README at the bottom of this module)
SAMPLE_RATE = 16_000
TARGET_RMS = 3000   # speech RMS after normalization (~ -21 dBFS)
TARGET_PEAK = 16000  # peak ceiling, ~ -6 dBFS
ELEVEN_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
ELEVEN_VOICES = {  # ElevenLabs premade voices; pass a raw voice id for anything else
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Adam": "pNInz6obpgDQGcFmaJgB",
}

# Characters actually sent to a provider (cache misses only), for cost tracking.
chars_rendered: dict[str, int] = {"polly": 0, "eleven": 0, "kokoro": 0, "piper": 0, "sapi": 0}


def _cache_path(spec: str, text: str) -> Path:
    key = hashlib.sha1(f"{spec}|{ELEVEN_MODEL}|{text}".encode()).hexdigest()[:16]
    return CACHE / f"{spec.replace(':', '_')}_{key}.pcm"


def _polly(voice: str, engine: str, text: str) -> bytes:
    import boto3

    client = boto3.client(
        "polly",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
    )
    resp = client.synthesize_speech(
        Text=text, VoiceId=voice, Engine=engine, OutputFormat="pcm", SampleRate=str(SAMPLE_RATE)
    )
    return resp["AudioStream"].read()


def _eleven(voice: str, text: str) -> bytes:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    voice_id = ELEVEN_VOICES.get(voice, voice)
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        params={"output_format": f"pcm_{SAMPLE_RATE}"},
        headers={"xi-api-key": key},
        json={"text": text, "model_id": ELEVEN_MODEL},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


_local: dict = {}  # loaded local models, reused across clips


def _to_16k(x: np.ndarray, rate: int) -> bytes:
    """float32 [-1, 1] or int16 samples at `rate` -> raw PCM16 mono 16 kHz."""
    import soxr

    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    if rate != SAMPLE_RATE:
        x = soxr.resample(x, rate, SAMPLE_RATE)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _kokoro(voice: str, text: str) -> bytes:
    if "kokoro" not in _local:
        from kokoro_onnx import Kokoro

        _local["kokoro"] = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    lang = "en-gb" if voice[:2] in ("bf", "bm") else "en-us"
    samples, rate = _local["kokoro"].create(text, voice=voice, speed=1.0, lang=lang)
    return _to_16k(samples, rate)


def _piper(voice: str, text: str) -> bytes:
    if voice not in _local:
        from piper import PiperVoice

        _local[voice] = PiperVoice.load(MODELS / f"{voice}.onnx")
    chunks = list(_local[voice].synthesize(text))
    if not chunks:
        raise RuntimeError("piper produced no audio")
    x = np.concatenate([c.audio_int16_array for c in chunks])
    return _to_16k(x, chunks[0].sample_rate)


def _sapi(voice: str, text: str) -> bytes:
    import subprocess
    import tempfile
    import wave

    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -like '*{voice}*' }} | Select-Object -First 1;"
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) } else { Write-Error 'voice not installed'; exit 1 };"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
        "$s.SetOutputToWaveFile($env:SAPI_OUT, $f); $s.Speak($env:SAPI_TEXT); $s.Dispose()"
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.wav"
        env = {**os.environ, "SAPI_OUT": str(out), "SAPI_TEXT": text}
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, check=True, capture_output=True)
        with wave.open(str(out)) as w:
            return w.readframes(w.getnframes())


def available(spec: str) -> str | None:
    """Returns a reason the voice can't be used, or None if it looks usable."""
    if spec.startswith("clip:"):
        return None  # a missing clip is reported per scenario when it is looked up
    if spec.startswith("eleven:") and not os.getenv("ELEVENLABS_API_KEY"):
        return "ELEVENLABS_API_KEY not set"
    if spec.startswith("kokoro:") and not (MODELS / "kokoro-v1.0.onnx").exists():
        return "evals/models/kokoro-v1.0.onnx + voices-v1.0.bin missing (see tts.py docstring)"
    if spec.startswith("piper:") and not (MODELS / f"{spec.partition(':')[2]}.onnx").exists():
        return f"evals/models/{spec.partition(':')[2]}.onnx missing (python -m piper.download_voices --download-dir evals/models <voice>)"
    if spec.startswith("polly:") and not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        return "AWS credentials not set"
    return None


def _normalize(pcm: bytes) -> bytes:
    """Scale to a common speech level so loud voices (ElevenLabs Adam peaks near full scale) don't distort."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    speech = x[np.abs(x) > 300]
    if speech.size == 0:
        return pcm
    gain = min(TARGET_RMS / np.sqrt(np.mean(speech**2)), TARGET_PEAK / max(np.abs(x).max(), 1.0))
    return np.clip(x * gain, -32768, 32767).astype("<i2").tobytes()


def synthesize(spec: str, text: str) -> bytes:
    """Render `text` with the voice `spec` (level-normalized), using the cache when possible.
    The cache holds the provider's raw output; normalization is applied on the way out."""
    if spec.startswith("clip:"):  # stored, checksummed recording: replayed exactly as saved (see clips.py)
        from . import clips

        return clips.get(spec[5:], text)
    return _normalize(_render(spec, text))


def _render(spec: str, text: str) -> bytes:
    path = _cache_path(spec, text)
    if path.exists():
        return path.read_bytes()
    provider, _, rest = spec.partition(":")
    if provider == "polly":
        voice, _, engine = rest.partition(":")
        pcm = _polly(voice, engine or "neural", text)
    elif provider == "eleven":
        pcm = _eleven(rest, text)
    elif provider == "kokoro":
        pcm = _kokoro(rest, text)
    elif provider == "piper":
        pcm = _piper(rest, text)
    elif provider == "sapi":
        pcm = _sapi(rest, text)
    else:
        raise ValueError(f"unknown voice spec {spec!r}")
    chars_rendered[provider] += len(text)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pcm)
    return pcm
