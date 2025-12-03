@echo off
echo ========================================
echo SignSpeak Data Collection Helper
echo ========================================
echo.
echo This script helps you collect training data for multiple signs.
echo.
echo Instructions:
echo   1. Enter the sign label (e.g., hello, thanks, yes)
echo   2. Enter number of sequences to collect (recommended: 20-30)
echo   3. Perform the sign when prompted
echo   4. Repeat for each sign you want to train
echo.
echo Press Ctrl+C to exit at any time
echo.

cd ml_training

:collect_loop
echo.
set /p SIGN_LABEL="Enter sign label (or 'done' to finish): "
if /i "%SIGN_LABEL%"=="done" goto :end

set /p NUM_SEQ="Enter number of sequences (default 20): "
if "%NUM_SEQ%"=="" set NUM_SEQ=20

echo.
echo Collecting %NUM_SEQ% sequences for sign: %SIGN_LABEL%
echo.
REM Try Python 3.11 first, fallback to default
py -3.11 collect_data.py --output_dir data --label %SIGN_LABEL% --num_sequences %NUM_SEQ% 2>nul
if %ERRORLEVEL% NEQ 0 (
    py collect_data.py --output_dir data --label %SIGN_LABEL% --num_sequences %NUM_SEQ%
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Successfully collected data for: %SIGN_LABEL%
) else (
    echo.
    echo ❌ Collection failed for: %SIGN_LABEL%
)

goto :collect_loop

:end
echo.
echo ========================================
echo Data collection complete!
echo.
echo Next step: Train the model
echo   Run: train_model.bat
echo ========================================
pause

