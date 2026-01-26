@echo off
echo Starting PrapAI Realtime Transcribe Server...
echo Open http://localhost:8000 in your browser
python -m uvicorn app:app --reload
pause
