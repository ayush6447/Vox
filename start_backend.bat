@echo off
echo ========================================
echo Starting SignSpeak Backend Server
echo ========================================
echo.
echo Backend will run on: http://localhost:8000
echo Press Ctrl+C to stop
echo.

cd backend
echo.
echo Choose backend:
echo   1. PyTorch (works with Python 3.14+) - RECOMMENDED
echo   2. TensorFlow (requires Python 3.8-3.11)
echo.
set /p BACKEND="Enter choice (1 or 2, default 1): "
if "%BACKEND%"=="" set BACKEND=1
if "%BACKEND%"=="2" (
    echo Starting TensorFlow backend...
    py -3.11 -m uvicorn main:app --reload --port 8000 2>nul
    if %ERRORLEVEL% NEQ 0 (
        py -m uvicorn main:app --reload --port 8000
    )
) else (
    echo Starting PyTorch backend...
    py -m uvicorn main_pytorch:app --reload --port 8000
)

pause

