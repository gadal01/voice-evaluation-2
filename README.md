# voice_sim_lab

One folder for testing the Nova Sonic voice agent with a simulated caller.

```
voice_sim_lab/
  voice_agent/     the agent under test: FastAPI + Nova Sonic + MCP tools, WebSocket at /ws (port 8000)
  simulator/       the simulated caller
    persona_agent/   base-agent config for the persona LLM (config.yaml, prompts, schema, sample request)
    sim_runner.py    runs one call: persona -> clip or Piper audio -> agent -> repeat -> judge
    clips.py tts.py  stored caller clips (caller_clips/) and live TTS (Piper only)
    personas.json    who calls and why;  scenarios.json  scripted scenarios and expectations
  ws_client/       WebSocket client: auth token, real-time PCM streaming, event capture, call recording
  judge/           LLM judge: llm_judge.py (rubric, Ollama or Bedrock), judge_cli.py (judge saved runs)
  live_console/    web page showing simulator | agent under test | judge, with live audio (port 8100)
  results/         one folder per run: JSON, transcript data, stereo WAV recording
```

## Setup (once)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2:7b                       # local model used for persona and judge
python -m piper.download_voices --download-dir simulator\models en_US-lessac-medium   # already copied if present
```
Copy `.env.example` to `.env` and fill in the AWS keys, `JWT_SECRET` and `AUTH_API_KEY` (a `.env` is already here).

## Run (from this folder)
```powershell
.\start_agent.ps1        # terminal 1: voice agent on :8000
.\start_console.ps1      # terminal 2: live console on :8100, open http://127.0.0.1:8100
```
No browser: `python -m simulator.sim_runner --persona sam_ac_impatient --persona-ollama ollama:qwen2:7b`
Judge saved runs: `python -m judge.judge_cli results\<run-folder>`
Stored clips: `python -m simulator.clips list|verify|record`

## Notes
- The persona runs on a local Ollama model by default. To use the base-agent persona-caller instead, pass
  `--persona-url http://<host>/persona-caller/turn` (see `simulator/persona_agent/README.md`).
- `voice_agent/` is a copy without the QA console. The original project is unchanged next to this folder.
- `aws_sdk_bedrock_runtime` is pinned below 0.13 because `sonic.py` uses the older client class name.
