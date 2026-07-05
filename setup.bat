@echo off
REM ============================================================================
REM TOMATO DISEASE DETECTION PROJECT - QUICK SETUP SCRIPT
REM ============================================================================
REM This script sets up the entire project environment
REM Run as: setup.bat
REM ============================================================================

echo.
echo ============================================================================
echo    TOMATO DISEASE DETECTION SYSTEM - SETUP WIZARD
echo ============================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [✓] Python found
python --version

echo.
echo ============================================================================
echo Step 1: Checking existing virtual environment...
echo ============================================================================
if exist venv (
    echo [✓] Virtual environment folder exists
    echo Do you want to use existing venv? (y/n)
    set /p use_existing=
    if /i not "%use_existing%"=="y" (
        echo [*] Removing old venv...
        rmdir /s /q venv
        echo [✓] Old venv removed
    ) else (
        goto activate_venv
    )
)

echo [*] Creating new virtual environment...
python -m venv venv
echo [✓] Virtual environment created

:activate_venv
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
echo [✓] Virtual environment activated

echo.
echo ============================================================================
echo Step 2: Upgrading pip, setuptools, and wheel...
echo ============================================================================
python -m pip install --upgrade pip setuptools wheel
echo [✓] Pip upgraded

echo.
echo ============================================================================
echo Step 3: Installing core dependencies (PyTorch + CUDA support)...
echo ============================================================================
echo.
echo Choose installation mode:
echo 1) GPU Support (CUDA 11.8) - Recommended for faster training
echo 2) CPU Only - Works but slower
echo.
set /p choice=Enter choice (1 or 2):

if "%choice%"=="1" (
    echo [*] Installing PyTorch with CUDA 11.8 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [*] Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)
echo [✓] PyTorch installed

echo.
echo ============================================================================
echo Step 4: Installing all project dependencies...
echo ============================================================================
echo [*] Installing from requirements.txt...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo [✓] All dependencies installed successfully!
) else (
    echo [!] Some packages failed to install, but continuing...
)

echo.
echo ============================================================================
echo Step 5: Installing missing critical packages...
echo ============================================================================
echo [*] Installing pytorch-grad-cam...
pip install pytorch-grad-cam
echo [*] Installing streamlit-webrtc...
pip install streamlit-webrtc
echo [*] Installing scikit-learn...
pip install scikit-learn
echo [*] Installing pyserial...
pip install pyserial
echo [✓] Critical packages installed

echo.
echo ============================================================================
echo Step 6: Verifying installation...
echo ============================================================================
echo.
echo Checking core packages...
python -c "import torch; print('✓ PyTorch', torch.__version__)" >nul 2>&1 && echo [✓] PyTorch || echo [!] PyTorch failed
python -c "import cv2; print('✓ OpenCV', cv2.__version__)" >nul 2>&1 && echo [✓] OpenCV || echo [!] OpenCV failed
python -c "import numpy; print('✓ NumPy', numpy.__version__)" >nul 2>&1 && echo [✓] NumPy || echo [!] NumPy failed
python -c "from ultralytics import YOLO; print('✓ YOLO loaded')" >nul 2>&1 && echo [✓] YOLO || echo [!] YOLO failed
python -c "from pytorch_grad_cam import GradCAM; print('✓ Grad-CAM loaded')" >nul 2>&1 && echo [✓] Grad-CAM || echo [!] Grad-CAM failed
python -c "import streamlit; print('✓ Streamlit loaded')" >nul 2>&1 && echo [✓] Streamlit || echo [!] Streamlit failed
python -c "import streamlit_webrtc; print('✓ WebRTC loaded')" >nul 2>&1 && echo [✓] Streamlit WebRTC || echo [!] Streamlit WebRTC failed

echo.
echo ============================================================================
echo Step 7: Checking model files...
echo ============================================================================
if exist "tomato_disease_model.pth" (
    echo [✓] tomato_disease_model.pth found
) else (
    echo [!] tomato_disease_model.pth NOT FOUND - Place in project root
)

if exist "classes.pth" (
    echo [✓] classes.pth found
) else (
    echo [!] classes.pth NOT FOUND - Place in project root
)

if exist "best_leaf_detector.pt" (
    echo [✓] best_leaf_detector.pt found
) else (
    echo [!] best_leaf_detector.pt NOT FOUND - Place in project root
)

echo.
echo ============================================================================
echo SETUP COMPLETE!
echo ============================================================================
echo.
echo Next steps:
echo   1. Verify all model files are in the project root directory
echo   2. Run the Streamlit app: streamlit run app.py
echo   3. Or test with CLI: python tomato.py --image test_image.jpg
echo.
echo To activate venv in future sessions:
echo   Windows: venv\Scripts\activate.bat
echo   Linux:   source venv/bin/activate
echo.
echo.
pause
