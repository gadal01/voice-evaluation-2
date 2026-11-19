# persona-caller (base-agent)

Stateless, one call per caller turn. The runner (WebSocket + Piper) sends persona, scenario, transcript and available
static clips; the agent returns the next caller line plus behavior hints.

    POST /persona-caller/turn   body: samples/request.json   -> turn_output.json shape

## Runner contract
`evals/sim_runner.py` builds this request (`build_request`): persona facts, transcript and clip catalog are pre-formatted
to text, as in samples/request.json. `--persona-ollama ollama:qwen2:7b` runs these same prompts locally without base-agent.

- `clip_id` set  -> play the stored clip (`evals/clips.py` lookup, static).
- `clip_id` null -> render `utterance` with Piper (`evals/tts.py`, `piper:<voice>`, dynamic, disk-cached).
- Wait `hesitation_ms` before sending audio; if `barge_in`, start speaking while the agent is still talking.
- `end_call` true -> send final audio, then close the socket.
- Hard-stop at `max_turns` even if the model never ends the call. On a schema or HTTP failure, fall back to the
  scenario's `dialogue.fallback` line from `evals/dialogue.py`.

## Assumptions to confirm against the real base-agent loader
- `{{data.x}}` prompt variable syntax (may be `{data.x}` or Jinja); structured values (facts, transcript, catalog) may
  need pre-formatting to text by the runner.
- Top-level config keys beyond those in the sample agent (`mcp.servers: []`, `skills.enabled: false`).
- Whether `schema` is JSON Schema draft-07 and relative to the agent folder.
- Whether the request envelope `{user, data}` passes `data` untouched into the prompt context.
- Redis cache key: should include the full request; if it only keys on a subset, disable it for this agent.
