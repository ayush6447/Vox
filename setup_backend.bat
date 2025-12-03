@echo off
echo ========================================
echo SignSpeak Backend Setup
echo ========================================
echo.

echo Checking Python version...
py --version
echo.
echo NOTE: TensorFlow requires Python 3.8-3.11
echo If you see Python 3.12+, please install Python 3.11
echo.

cd backend
echo Installing Python dependencies...
echo.
echo Choose framework:
echo   1. PyTorch (works with Python 3.14+) - RECOMMENDED
echo   2. TensorFlow (requires Python 3.8-3.11)
echo.
set /p FRAMEWORK="Enter choice (1 or 2, default 1): "
if "%FRAMEWORK%"=="" set FRAMEWORK=1
if "%FRAMEWORK%"=="2" (
    echo Installing TensorFlow dependencies...
    py -3.11 -m pip install -r requirements.txt 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo Trying default Python...
        py -m pip install -r requirements.txt
    )
) else (
    echo Installing PyTorch dependencies...
    py -m pip install -r requirements_pytorch.txt
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Backend dependencies installed successfully!
    echo.
    echo To start the backend server, run:
    echo   cd backend
    echo   uvicorn main:app --reload --port 8000
) else (
    echo.
    echo ❌ Installation failed. Please check your Python installation.
)

pause

