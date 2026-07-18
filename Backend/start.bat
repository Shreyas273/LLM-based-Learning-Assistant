@echo off
echo Starting AI Learning Assistant Backend...
echo.
echo Make sure you have:
echo 1. Set your GEMINI_API_KEY in .env file
echo 2. Installed all dependencies: pip install -r requirements.txt
echo.
echo Backend will start on: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
