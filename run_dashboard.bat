@echo off
title SUS Controller - 1-Click Launch
echo ========================================================
echo    SUS: Streaming Uncertainty Surveillance Framework
echo ========================================================
echo.

cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [SETUP] First time setup detected. Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv!
        pause
        exit /b 1
    )
    echo [SETUP] Installing required dependencies (Flask, NumPy, Scikit-Learn, SciPy, Pandas)...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install packages!
        pause
        exit /b 1
    )
    echo [SETUP] Setup complete!
    echo.
)

echo [STARTING] Launching SUS Demo Server at http://localhost:5000 ...
echo [INFO] Please open your browser at: http://localhost:5000
echo [INFO] Press Ctrl+C in this window when you want to stop the server.
echo.

:: Automatically open browser after 2 seconds
start "" http://localhost:5000

:: Run Flask App
venv\Scripts\python.exe demo\app.py
pause

