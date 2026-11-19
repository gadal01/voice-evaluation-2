"""Thin wrapper around an Amazon Nova Sonic bidirectional stream on Bedrock.

One `SonicSession` == one Bedrock stream == one conversation.
Nova Sonic does its own VAD, STT, LLM and TTS, so we just forward mic audio in
and surface events out.
"""
import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator

from aws_sdk_bedrock_runtime.client import (
    AsyncBedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity import AWSCredentialsIdentity, StaticCredentialsResolver
from smithy_http.aio.crt import AWSCRTHTTPClient

from agent_settings import get_settings

# Written to AWS's Nova 2 Sonic prompting guidance: a short baseline, direct answers (no reasoning
# aloud) and a one-shot example rather than lists of phrases, which Sonic 2 over-uses.
DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, professional, and helpful AI assistant for a service desk, talking with a caller "
    "on the phone. Give accurate answers that sound natural, direct, and human. Answer in 1-2 sentences "
    "and never more than 3-5 short sentences. Everything you say is spoken aloud, so say only what you "
    "would say to the caller. Tools work quietly in the background; the caller only hears the answer, "
    "so for simple questions and for tool results give the answer directly with no explanation of how "
    "you got it. "
    "Use your tools to look up serviceable items, quote servicing costs and appointment slots, and book "
    "service requests. Never guess prices, item ids or slots. When searching items, do not filter by "
    "category unless the caller asked for one. To book a request, follow these steps in order. First, make sure you have the caller's "
    "name, phone number, item, brand (the make, for example Samsung), the symptom (what is wrong) and slot. Second, read the phone number back one digit at a time, together with the quoted total, and ask the caller to confirm. Third, only after a clear yes, create the request. Finally, "
    "read out the request id.\n\n"
    "Example:\n"
    "Caller: How much is a repair on the washing machine?\n"
    "Assistant: A standard repair on the washing machine comes to <total> dollars, including parts and "
    "tax. Would you like me to book it?\n\n"
    "Example of the confirmation step:\n"
    "Assistant: I have your number as <phone digits, one at a time>. The inspection for the <item> comes to "
    "<total> dollars, with the earliest slot on <date>. Is all of that correct?"
)

IN_RATE = 16_000
OUT_RATE = 24_000


class SonicSession:
    def __init__(self) -> None:
        self.prompt_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self._stream = None

    # ------------------------------------------------------------ sending ---
    async def _send(self, event: dict) -> None:
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps({"event": event}).encode())
        )
        await self._stream.input_stream.send(chunk)

    async def start(self, tools: list[dict] | None = None) -> None:
        """`tools`: [{name, description, input_schema(dict)}] advertised to the model."""
        s = get_settings()
        self._settings = s
        # This SDK version does not pick a credential source on its own, so hand it the credentials from the
        # settings explicitly (they may come from .env, which is not copied into os.environ). Without a
        # resolver the SDK hangs silently.
        if not (s.aws_access_key_id and s.aws_secret_access_key):
            raise RuntimeError(
                "AWS credentials missing: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "(plus AWS_SESSION_TOKEN for temporary creds) in .env, then restart."
            )
        identity = AWSCredentialsIdentity(access_key_id=s.aws_access_key_id, secret_access_key=s.aws_secret_access_key,
                                          session_token=s.aws_session_token)
        # Bidirectional streaming needs the CRT transport (the default aiohttp one can't).
        config = await AsyncBedrockRuntimeConfig.resolve(
            region=s.aws_region,
            transport=AWSCRTHTTPClient(),
            aws_credentials_identity_resolver=StaticCredentialsResolver(identity=identity),
        )
        client = AsyncBedrockRuntimeClient(config=config)
        self._stream = await asyncio.wait_for(
            client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=s.nova_sonic_model_id)
            ),
            timeout=20,
        )

        # Auth/connection errors surface on the output future, not on send(), so race them
        # or a bad credential setup would hang here forever.
        first = asyncio.ensure_future(self._send({"sessionStart": {"inferenceConfiguration": {
            "maxTokens": 1024, "topP": 0.9, "temperature": 0.7}}}))
        out = asyncio.ensure_future(self._stream.await_output())
        done, _ = await asyncio.wait({first, out}, timeout=20, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            raise TimeoutError("Timed out connecting to Bedrock (check AWS credentials/region)")
        for task in done:
            task.result()  # re-raises credential / access errors
        await first
        prompt_start = {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": OUT_RATE, "sampleSizeBits": 16,
                "channelCount": 1, "voiceId": s.nova_sonic_voice, "encoding": "base64", "audioType": "SPEECH"},
        }
        if tools:
            prompt_start["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt_start["toolConfiguration"] = {"tools": [
                {"toolSpec": {
                    "name": t["name"], "description": t["description"],
                    # Sonic wants the JSON schema as a *string*, not an object.
                    "inputSchema": {"json": json.dumps(t["input_schema"])}}}
                for t in tools]}
        await self._send({"promptStart": prompt_start})

        # System prompt as a single TEXT content block.
        sys_name = str(uuid.uuid4())
        await self._send({"contentStart": {
            "promptName": self.prompt_name, "contentName": sys_name, "type": "TEXT",
            "interactive": False, "role": "SYSTEM",
            "textInputConfiguration": {"mediaType": "text/plain"}}})
        await self._send({"textInput": {
            "promptName": self.prompt_name, "contentName": sys_name, "content": s.nova_sonic_prompt or DEFAULT_SYSTEM_PROMPT}})
        await self._send({"contentEnd": {"promptName": self.prompt_name, "contentName": sys_name}})

        # Open the (long-lived) audio input block; mic frames are appended to it.
        await self._send({"contentStart": {
            "promptName": self.prompt_name, "contentName": self.audio_content_name,
            "type": "AUDIO", "interactive": True, "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": IN_RATE, "sampleSizeBits": 16,
                "channelCount": 1, "audioType": "SPEECH", "encoding": "base64"}}})

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send({"audioInput": {
            "promptName": self.prompt_name, "contentName": self.audio_content_name,
            "content": base64.b64encode(pcm16).decode()}})

    async def send_tool_result(self, tool_use_id: str, result: object) -> None:
        name = str(uuid.uuid4())
        await self._send({"contentStart": {
            "promptName": self.prompt_name, "contentName": name, "type": "TOOL",
            "interactive": False, "role": "TOOL",
            "toolResultInputConfiguration": {
                "toolUseId": tool_use_id, "type": "TEXT",
                "textInputConfiguration": {"mediaType": "text/plain"}}}})
        await self._send({"toolResult": {
            "promptName": self.prompt_name, "contentName": name,
            # Sonic only accepts a JSON *object* as the tool result, so wrap lists/strings.
            "content": json.dumps(result if isinstance(result, dict) else {"result": result})}})
        await self._send({"contentEnd": {"promptName": self.prompt_name, "contentName": name}})

    async def close(self) -> None:
        if not self._stream:
            return
        try:
            await self._send({"contentEnd": {
                "promptName": self.prompt_name, "contentName": self.audio_content_name}})
            await self._send({"promptEnd": {"promptName": self.prompt_name}})
            await self._send({"sessionEnd": {}})
            await self._stream.input_stream.close()
        except Exception:
            pass  # stream may already be gone

    # ---------------------------------------------------------- receiving ---
    async def events(self) -> AsyncIterator[tuple[str, object]]:
        """Yields (kind, payload):
        ("audio", bytes) | ("user_text", str) | ("assistant_text", str) | ("interrupted", None)
        | ("tool_use", {"id", "name", "args"})
        """
        _, output = await self._stream.await_output()
        stage = None  # generationStage of the current content block
        role = None
        tool: dict | None = None  # toolUse payload, emitted once its content block ends
        async for raw in output:
            data = getattr(getattr(raw, "value", None), "bytes_", None)
            if not data:
                continue
            ev = json.loads(data.decode()).get("event", {})

            if "contentStart" in ev:
                cs = ev["contentStart"]
                role = cs.get("role")
                fields = cs.get("additionalModelFields")
                stage = json.loads(fields).get("generationStage") if fields else None
            elif "toolUse" in ev:
                tu = ev["toolUse"]
                content = tu.get("content") or "{}"
                tool = {"id": tu["toolUseId"], "name": tu["toolName"],
                        "args": json.loads(content) if isinstance(content, str) else content}
            elif "contentEnd" in ev and ev["contentEnd"].get("type") == "TOOL" and tool:
                yield "tool_use", tool
                tool = None
            elif "audioOutput" in ev:
                yield "audio", base64.b64decode(ev["audioOutput"]["content"])
            elif "textOutput" in ev:
                text = ev["textOutput"]["content"]
                if '"interrupted"' in text:   # barge-in marker is sent as JSON text
                    yield "interrupted", None
                elif role == "USER":
                    yield "user_text", text
                elif stage != "SPECULATIVE":  # SPECULATIVE text is repeated as FINAL
                    yield "assistant_text", text
            elif "contentEnd" in ev and ev["contentEnd"].get("stopReason") == "INTERRUPTED":
                yield "interrupted", None
