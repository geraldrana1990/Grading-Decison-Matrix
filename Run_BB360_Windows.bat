@echo off
REM === BB360 one-click launcher (Windows .bat) ===
REM 1) Double-click this file.
REM 2) It will create a virtual env, install deps, and start the app in your browser.

setlocal
cd /d %~dp0

REM Detect Python
where python >nul 2>&1
if %errorlevel% neq 0 (
  where py >nul 2>&1
  if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.10+ from python.org and re-run.
    pause
    exit /b 1
  ) else (
    set PY=py
  )
) else (
  set PY=python
)

REM Create venv if missing
if not exist .venv (
  %PY% -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip (quiet) and install requirements
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

REM Launch Streamlit
start "" http://localhost:8501/
streamlit run BB360_app.py --server.port=8501 --server.address=0.0.0.0

REM Keep window open on exit
echo.
echo App stopped. Press any key to close this window.
pause >nul
