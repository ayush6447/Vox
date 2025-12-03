@echo off
echo ========================================
echo SignSpeak Model Training
echo ========================================
echo.

cd ml_training

if not exist "data" (
    echo ⚠️  No data directory found!
    echo.
    echo Please collect training data first:
    echo   py collect_data.py --output_dir data --label hello --num_sequences 20
    echo   py collect_data.py --output_dir data --label thanks --num_sequences 20
    echo.
    pause
    exit /b
)

echo Training model with data from: data/
echo.
echo Choose framework:
echo   1. PyTorch (works with Python 3.14+) - RECOMMENDED
echo   2. TensorFlow (requires Python 3.8-3.11)
echo.
set /p FRAMEWORK="Enter choice (1 or 2, default 1): "
if "%FRAMEWORK%"=="" set FRAMEWORK=1
if "%FRAMEWORK%"=="2" (
    echo Using TensorFlow...
    py -3.11 train_model.py --data_dir data --output_model ../backend/model/sign_model.h5 --epochs 25 2>nul
    if %ERRORLEVEL% NEQ 0 (
        py train_model.py --data_dir data --output_model ../backend/model/sign_model.h5 --epochs 25
    )
) else (
    echo Using PyTorch...
    py train_model_pytorch.py --data_dir data --output_model ../backend/model/sign_model.pt --epochs 25
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Model trained successfully!
    echo Model saved to: backend/model/sign_model.h5
) else (
    echo.
    echo ❌ Training failed. Please check the error messages above.
)

pause

