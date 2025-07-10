@echo off
echo Activating RAG Document System virtual environment...
call ragvenv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo You can now run:
echo   python main.py --help
echo   streamlit run advanced_chat.py
echo   python model_manager.py
echo.
cmd /k 