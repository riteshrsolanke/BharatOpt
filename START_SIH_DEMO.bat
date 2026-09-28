@echo off
setlocal
TITLE BharatOpt-X: Sovereign GPU-Accelerated Optimization Engine

echo ==========================================================
echo       BHARATOPT-X - SMART INDIA HACKATHON 2026
echo             TEAM NEXORA - PS ID: SIH26119
echo ==========================================================
echo.
echo [1/3] Checking dependencies...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.10+
    pause
    exit /b
)

echo [2/3] Installing/Verifying Python requirements...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install some dependencies. The server might still run if already installed.
)

:: Check if C++ engine is built, if not try to build it
if not exist "bharatopt_engine.exe" (
    if not exist "build\bharatopt_engine.exe" (
        where cmake >nul 2>&1
        if %errorlevel% equ 0 (
            echo [AUTO-BUILD] Compiling native C++ engine...
            cmake -B build -DCMAKE_BUILD_TYPE=Release >nul 2>&1
            cmake --build build --config Release >nul 2>&1
        )
    )
)

:: Free port 8000 if an older server is already running
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

echo [3/3] Launching BharatOpt-X Core Engine ^& AI Sidecar...
echo.
echo ==========================================================
echo The Dashboard will open at: http://localhost:8000
echo Leave this terminal open. Press Ctrl+C to stop the server.
echo ==========================================================
echo.

:: Give the server 2 seconds to start, then open the browser
start "" http://localhost:8000

:: Start the FastAPI server
python services/api/main.py

pause
