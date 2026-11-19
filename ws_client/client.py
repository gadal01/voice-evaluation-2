"""WebSocket client for the Nova Sonic voice agent: one live conversation (auth, real-time PCM streaming, event capture).

Used by the simulator (simulator/sim_runner.py) and the live console. Protocol: see ../main.py.
  client -> server  binary : raw PCM16 mono 16 kHz, streamed in 30 ms frames (silence when the caller is quiet)
  server -> client  text   : JSON events ready | transcript | reply | interrupted | tool | error
  server -> client  binary : raw PCM16 mono 24 kHz agent speech
"""
import asyncio
import collections
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets

FRAME_BYTES, FRAME_S = 960, 0.03          # 30 ms of 16 kHz PCM16
OUT_BYTES_PER_S = 24_000 * 2              # agent audio is 24 kHz PCM16
SETTLE_S = 3.0                            # quiet time that means "the agent finished this turn"
TURN_TIMEOUT_S = 60


# ------------------------------------------------------------ one live conversation ---
class Conversation:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list[dict] = []
        self.turn = -1
        self.ready = asyncio.Event()
        self.error: str | None = None
        self.last_activity = 0.0
        self.last_audio = 0.0
        self.audio_secs: dict[int, float] = collections.defaultdict(float)
        self.first_audio: dict[int, float] = {}
        self.user_start: dict[int, float] = {}
        self.user_end: dict[int, float] = {}
        self.early_audio: dict[int, float] = {}  # agent audio that began while the caller was still talking
        self.said: list[str] = []  # the caller lines actually spoken, in order (reactive scenarios vary per call)
        self.queue: collections.deque = collections.deque()
        self.caller_audio: list[tuple[float, bytes]] = []  # (time sent, 16 kHz PCM16 frame)
        self.agent_audio: list[tuple[float, bytes]] = []   # (time received, 24 kHz PCM16 chunk)
        self.on_audio = None  # optional callback(channel, pcm): channel 0 = caller 16 kHz, 1 = agent 24 kHz (live monitoring)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    async def receive(self, ws) -> None:
        try:
            async for msg in ws:
                t = self.now()
                if isinstance(msg, bytes):
                    self.last_activity = self.last_audio = t
                    if t >= self.user_end.get(self.turn, float("inf")):
                        self.first_audio.setdefault(self.turn, t)
                    elif self.turn in self.user_start:  # the agent started talking before the caller finished
                        self.early_audio.setdefault(self.turn, t)
                    self.audio_secs[self.turn] += len(msg) / OUT_BYTES_PER_S
                    self.agent_audio.append((t, msg))
                    if self.on_audio:
                        self.on_audio(1, msg)
                    continue
                ev = json.loads(msg)
                ev.update(t=t, turn=self.turn)
                self.events.append(ev)
                if ev["type"] == "ready":
                    self.ready.set()
                elif ev["type"] == "error":
                    self.error = ev["message"]
                    self.ready.set()
                else:
                    self.last_activity = t
        except websockets.ConnectionClosed:
            pass
        finally:
            self.ready.set()

    async def send_loop(self, ws) -> None:
        """Streams 30 ms frames at real time; silence whenever the caller isn't speaking."""
        nxt = time.perf_counter()
        silence = b"\x00" * FRAME_BYTES
        while True:
            from_queue = bool(self.queue)
            frame, done = self.queue.popleft() if from_queue else (silence, None)
            await ws.send(frame)
            if from_queue:
                self.caller_audio.append((self.now(), frame))
                if self.on_audio:
                    self.on_audio(0, frame)
            if done:
                done.set_result(self.now())
            nxt += FRAME_S
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    async def say(self, pcm: bytes) -> float:
        """Queue a clip and wait until its last frame has been sent; returns that time."""
        done = asyncio.get_running_loop().create_future()
        frames = [pcm[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00") for i in range(0, len(pcm), FRAME_BYTES)]
        for k, f in enumerate(frames):
            self.queue.append((f, done if k == len(frames) - 1 else None))
        return await done

    def turn_done(self, i: int) -> bool:
        t = self.now()
        tools = [e["t"] for e in self.events if e["turn"] == i and e["type"] == "tool"]
        spoke = i in self.first_audio or any(e["turn"] == i and e["type"] == "reply" for e in self.events)
        if not spoke or (tools and self.last_audio <= max(tools)):  # nothing yet, or a tool result not answered yet
            return False
        playback_end = self.first_audio.get(i, t) + self.audio_secs[i]
        return t >= max(self.last_activity + SETTLE_S, playback_end + 0.5)


async def get_token(ws_url: str) -> str:
    """Exchanges AUTH_API_KEY for a JWT at the agent's /auth/token endpoint (a fresh one per call, they are short-lived)."""
    import os

    import httpx

    base = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1).rsplit("/ws", 1)[0]
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{base}/auth/token", headers={"X-API-Key": os.getenv("AUTH_API_KEY", "")})
    if r.status_code != 200:
        raise RuntimeError(f"could not get a token from {base}/auth/token (HTTP {r.status_code}): set AUTH_API_KEY in .env")
    return r.json()["access_token"]


# -------------------------------------------------------------- audio export ---
def write_conversation_wav(conversation: Conversation, path: Path) -> float:
    """Stereo 16 kHz WAV on the call's real timeline: caller on the left, agent on the right.
    Agent chunks arrive faster than real time, so they are queued like a speaker would play them."""
    rate = 16_000
    caller = [(t, np.frombuffer(f, dtype="<i2")) for t, f in conversation.caller_audio]
    agent, cursor = [], 0.0
    for t, chunk in conversation.agent_audio:
        cursor = max(cursor, t)
        x = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype="<i2").astype(np.float64)
        n_out = int(len(x) * rate / 24_000)  # 24 kHz -> 16 kHz (linear; fine for listening back)
        agent.append((cursor, np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype("<i2") if len(x) > 1 else x.astype("<i2")))
        cursor += len(x) / 24_000
    end = max([t + len(x) / rate for t, x in caller] + [t + len(x) / rate for t, x in agent] + [0.0]) + 0.5
    left, right = np.zeros(int(end * rate), "<i2"), np.zeros(int(end * rate), "<i2")
    for track, items in ((left, caller), (right, agent)):
        for t, x in items:
            i = int(t * rate)
            track[i:i + len(x)] = x[: len(track) - i]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.column_stack([left, right]).tobytes())
    return end
