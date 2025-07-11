@echo off
echo Starting RAG Document Search Chat Interface...
echo.

REM Activate the virtual environment
call ragvenv\Scripts\activate.bat

REM Start the advanced chat interface
echo Starting Streamlit interface...
streamlit run advanced_chat.py

pause 