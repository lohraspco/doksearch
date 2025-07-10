@echo off
echo Restarting RAG Chat App...
echo.

REM Kill any existing Streamlit processes
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

REM Start the app with logging
python restart_app.py

pause 