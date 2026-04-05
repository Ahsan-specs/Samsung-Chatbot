@echo off
echo =========================================================
echo   Samsung AI Support Assistant - Setup ^& Launch
echo   Agentic RAG ^| Multi-Modal ^| Groq LLaMA 3.3 70B
echo =========================================================
echo.

echo [1/5] Setting up Python Virtual Environment...
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    echo     Virtual environment created.
) else (
    echo     Virtual environment already exists.
)

echo [2/5] Activating Virtual Environment...
call venv\Scripts\activate

echo [3/5] Installing Dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet 2>pip_error.log
if %errorlevel% neq 0 (
    echo     WARNING: Some packages may have failed. Check pip_error.log
)

echo [4/5] Downloading NLP Models...
python -m spacy download en_core_web_sm --quiet 2>nul

echo [5/5] Launching Samsung AI Support Assistant...
echo.
echo =========================================================
echo   App running at: http://localhost:8501
echo   Press Ctrl+C to stop the server.
echo =========================================================
echo.
streamlit run app.py --server.port 8501 --server.headless true

pause
