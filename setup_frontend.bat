@echo off
echo ========================================
echo SignSpeak Frontend Setup
echo ========================================
echo.

cd frontend
echo Installing Node.js dependencies...
call npm install

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Frontend dependencies installed successfully!
    echo.
    echo To start the frontend dev server, run:
    echo   cd frontend
    echo   npm run dev
) else (
    echo.
    echo ❌ Installation failed. Please check your Node.js installation.
)

pause


