"""Voice agent over a WebSocket (FastAPI + uvicorn) backed by Amazon Nova Sonic + MCP tools.

Protocol (client <-> server on /ws):
  client -> server  binary : raw PCM16 mono 16 kHz mic audio
  server -> client  text   : JSON events: ready | transcript | reply | interrupted | tool | error
  server -> client  binary : raw PCM16 mono 24 kHz agent speech

Nova Sonic handles voice-activity detection, STT, the LLM and TTS in one stream, including
barge-in. When the model calls a tool, we forward the call to an MCP server and feed the
result back. The sample MCP server (mcp_server.py) is mounted in this app at /mcp; point
MCP_SERVER_URL at any other streamable-HTTP MCP server to use that instead.
"""
import asyncio
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import auth
from agent_settings import get_settings
from mcp_server import mcp
from sonic import IN_RATE, OUT_RATE, SonicSession

log = logging.getLogger("voice")
MCP_URL = get_settings().mcp_server_url or f"http://127.0.0.1:{get_settings().port}/mcp/"


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        auth._secret()
    except auth.AuthError as exc:
        log.error("WebSocket auth is NOT configured, every connection will be refused: %s", exc)
    async with mcp.session_manager.run():  # required for the mounted streamable-HTTP app
        yield


app = FastAPI(title="Nova Sonic Voice Agent", lifespan=lifespan)
app.mount("/mcp", mcp.streamable_http_app())  # routes: /mcp/


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


# ---------------------------------------------------------------- MCP layer --
async def list_tools(mcp_session: ClientSession) -> list[dict]:
    result = await mcp_session.list_tools()
    return [{"name": t.name, "description": t.description or t.name, "input_schema": t.inputSchema}
            for t in result.tools]


async def call_tool(mcp_session: ClientSession, name: str, args: dict) -> object:
    try:
        res = await mcp_session.call_tool(name, args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
    if res.isError:
        return {"error": text or "tool failed"}
    if res.structuredContent:
        return res.structuredContent.get("result", res.structuredContent)  # list results are wrapped
    try:
        return json.loads(text)
    except ValueError:
        return text


async def handle_tool_use(ws: WebSocket, sonic: SonicSession, mcp_session: ClientSession, tool: dict) -> None:
    log.info("tool call %s %s", tool["name"], tool["args"])
    started = time.perf_counter()
    result = await call_tool(mcp_session, tool["name"], tool["args"])
    ms = round((time.perf_counter() - started) * 1000)
    await ws.send_json({"type": "tool", "name": tool["name"], "args": tool["args"], "result": result, "ms": ms})
    await sonic.send_tool_result(tool["id"], result)


async def relay_from_sonic(ws: WebSocket, sonic: SonicSession, mcp_session: ClientSession) -> None:
    pending: set[asyncio.Task] = set()  # tool calls run concurrently so audio keeps flowing
    try:
        async for kind, payload in sonic.events():
            if kind == "audio":
                await ws.send_bytes(payload)
            elif kind == "user_text":
                await ws.send_json({"type": "transcript", "text": payload})
            elif kind == "assistant_text":
                await ws.send_json({"type": "reply", "text": payload})
            elif kind == "interrupted":
                await ws.send_json({"type": "interrupted"})
            elif kind == "tool_use":
                task = asyncio.create_task(handle_tool_use(ws, sonic, mcp_session, payload))
                pending.add(task)
                task.add_done_callback(pending.discard)
    finally:
        for task in pending:
            task.cancel()


@app.post("/auth/token")
async def create_token(x_api_key: str | None = Header(default=None)) -> dict:
    """Exchange the API key for a short-lived JWT to open /ws with."""
    if not auth.check_api_key(x_api_key):
        await asyncio.sleep(0.5)  # slow down key guessing
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    try:
        token, ttl = auth.issue_token()
    except auth.AuthError as exc:
        log.error("cannot issue tokens: %s", exc)
        raise HTTPException(status_code=503, detail="authentication is not configured on the server")
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl}


@app.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    # Verify the JWT before accepting: a bad or missing token never gets a socket (handshake -> HTTP 403).
    try:
        claims, subprotocol = auth.authenticate(ws)
    except auth.AuthError as exc:
        log.warning("websocket rejected from %s: %s", ws.client.host if ws.client else "?", exc)
        await ws.close(code=1008)
        return
    # The server creates the conversation id on every successful auth; clients read it back from `ready`.
    conversation_id = uuid.uuid4().hex
    log.info("websocket accepted for %s, conversation %s", claims["sub"], conversation_id)
    await ws.accept(subprotocol=subprotocol)
    sonic = SonicSession()
    out_task: asyncio.Task | None = None
    try:
        async with contextlib.AsyncExitStack() as stack:
            read, write, _ = await stack.enter_async_context(streamablehttp_client(MCP_URL))
            mcp_session = await stack.enter_async_context(ClientSession(read, write))
            await mcp_session.initialize()
            tools = await list_tools(mcp_session)
            log.info("MCP tools: %s", [t["name"] for t in tools])

            await sonic.start(tools)
            out_task = asyncio.create_task(relay_from_sonic(ws, sonic, mcp_session))
            await ws.send_json({"type": "ready", "in_rate": IN_RATE, "out_rate": OUT_RATE,
                                "tools": [t["name"] for t in tools], "conversation_id": conversation_id})
            try:
                while True:
                    if out_task.done():  # Bedrock side ended (timeout, error, ...)
                        out_task.result()
                        break
                    try:
                        pcm = await asyncio.wait_for(ws.receive_bytes(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    await sonic.send_audio(pcm)
            finally:
                if out_task:
                    out_task.cancel()
                await sonic.close()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("voice session failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="127.0.0.1", port=get_settings().port,
                reload=False)
