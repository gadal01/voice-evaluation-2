"""LLM-as-judge for a finished call. Scores the whole conversation against a rubric via Bedrock Converse.

The judge sees the scenario goal, what the caller said, what the agent heard, the agent's replies and every
tool call with its real result, so it can catch what rule checks can't: invented prices or slots, a rambling or
unspeakable answer, a booking made without a clear yes. It is advisory: it never changes `passed`.
"""
import json
import os
import re

CRITERIA = {
    "task_success": "Did the agent accomplish what the caller wanted (or correctly decline when it is out of scope)?",
    "accuracy": "Is every price, slot, id, name and number the agent states supported by the tool results? 1 if anything is invented.",
    "policy": "Did it follow the rules: confirm details and price before booking, read back the phone number, never book unless the caller clearly agreed, stay in scope?",
    "voice_style": "Is it suited to speech: short, natural, no lists/markdown/ids read out, no reasoning or tool names spoken aloud?",
}
MODEL = os.getenv("JUDGE_MODEL", "us.amazon.nova-pro-v1:0")
PASS_AT = 4  # verdict is "pass" when every criterion scores at least this

SYSTEM = (
    "You are a strict evaluator of a voice assistant that books home-appliance servicing by phone. "
    "You are given one call: the scenario goal, the caller's words, what the assistant heard, its spoken replies, and "
    "its tool calls with real results. Judge only what is in the call. Score each criterion 1-5 "
    "(5 flawless, 4 minor nit, 3 noticeable problem, 2 serious, 1 failed or harmful). Speech-recognition slips by the "
    "assistant that it recovered from are not its fault. Asking \"would you like me to book that?\" is only an offer and "
    "is fine. A policy problem is only: actually booking without the caller's clear agreement, skipping the phone read-back "
    "before booking, or acting outside scope. Do not demand confirmations for read-only questions.\n\nCriteria:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    + '\n\nReply with ONLY a JSON object: {"scores": {' + ", ".join(f'"{k}": int' for k in CRITERIA)
    + '}, "issues": [short strings, empty if none], "reason": "one or two sentences"}'
)


def _render(scenario: dict, run: dict) -> str:
    lines = [f"Scenario goal: {scenario.get('description', scenario['id'])}", ""]
    for c in run.get("conversation", []):
        role = c["role"]
        if role == "caller":
            lines.append(f"CALLER SAID: {c['text']}")
        elif role == "heard":
            lines.append(f"  (assistant heard: {c['text']})")
        elif role == "agent":
            lines.append(f"ASSISTANT SAID: {c['text']}")
        elif role == "tool":
            lines.append(f"  [tool {c['name']} args={json.dumps(c['args'])} -> {json.dumps(c['result'])[:600]}]")
        else:
            lines.append(f"  [system: {c['text']}]")
    return "\n".join(lines)


def _client():
    import boto3

    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
    )


def _ollama(model: str, system: str, user: str) -> str:
    """Local judge via Ollama's chat API (model spec "ollama:qwen2:7b"). stdlib only; JSON mode, temperature 0."""
    import urllib.request

    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=int(os.getenv("OLLAMA_TIMEOUT", "180"))) as r:
        return json.loads(r.read())["message"]["content"]


def evaluate(scenario: dict, run: dict, model: str | None = None) -> dict:
    """Returns {"scores", "overall", "pass", "issues", "reason", "model"} or {"error", "model"}. Never raises.
    model "ollama:<name>" uses a local Ollama model; anything else is a Bedrock model id."""
    model = model or MODEL
    if not run.get("conversation"):
        return {"error": "no conversation to judge", "model": model}
    try:
        if model.startswith("ollama:"):
            text = _ollama(model[len("ollama:"):], SYSTEM, _render(scenario, run))
        else:
            resp = _client().converse(
                modelId=model,
                system=[{"text": SYSTEM}],
                messages=[{"role": "user", "content": [{"text": _render(scenario, run)}]}],
                inferenceConfig={"maxTokens": 600, "temperature": 0},
            )
            text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        scores = {k: max(1, min(5, int(data["scores"][k]))) for k in CRITERIA}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "model": model}
    return {
        "scores": scores, "overall": round(sum(scores.values()) / len(scores), 2),
        "pass": min(scores.values()) >= PASS_AT, "issues": [str(i) for i in data.get("issues", [])][:6],
        "reason": str(data.get("reason", "")).strip(), "model": model,
    }
