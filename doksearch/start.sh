#!/bin/bash

echo "========================================"
echo "     RAG System with Hybrid Retriever"
echo "========================================"
echo

echo "[1/4] Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[OK] Ollama is already running"
else
    echo "[INFO] Starting Ollama server..."
    ollama serve &
    echo "[INFO] Waiting for Ollama to start..."
    sleep 5
fi

echo
echo "[2/4] Checking for Gemma3:4b model..."
if ollama list | grep -q "gemma3:4b"; then
    echo "[OK] Gemma3:4b model is available"
else
    echo "[INFO] Downloading Gemma3:4b model (this may take a while)..."
    ollama pull gemma3:4b
fi

echo
echo "[3/4] Checking Python dependencies..."
if python3 -c "import streamlit" >/dev/null 2>&1; then
    echo "[OK] Dependencies are installed"
else
    echo "[INFO] Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

echo
echo "[4/4] Starting RAG Web UI..."
echo "[INFO] Opening web browser to http://localhost:8501"
echo "[INFO] Press Ctrl+C to stop the server"
echo

streamlit run web_ui.py 