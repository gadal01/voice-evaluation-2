# Voice agent under test on http://127.0.0.1:8000 (ws://127.0.0.1:8000/ws)
Set-Location "$PSScriptRoot\voice_agent"
python -m uvicorn main:app --port 8000
