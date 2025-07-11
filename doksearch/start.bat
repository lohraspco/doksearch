@echo off
echo ========================================
echo      RAG System with Hybrid Retriever
echo ========================================
echo.

echo [1/4] Checking if Ollama is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Starting Ollama server...
    start "Ollama Server" ollama serve
    echo [INFO] Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
) else (
    echo [OK] Ollama is already running
)

echo.
echo [2/4] Checking for Gemma3:4b model...
ollama list | findstr "gemma3:4b" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Downloading Gemma3:4b model (this may take a while)...
    ollama pull gemma3:4b
) else (
    echo [OK] Gemma3:4b model is available
)

echo.
echo [3/4] Checking Python dependencies...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing Python dependencies...
    pip install -r requirements.txt
) else (
    echo [OK] Dependencies are installed
)

echo.
echo [4/4] Starting RAG Web UI...
echo [INFO] Opening web browser to http://localhost:8501
echo [INFO] Press Ctrl+C to stop the server
echo.

streamlit run web_ui.py 