#!/bin/bash
# ============================================================================
# TOMATO DISEASE DETECTION PROJECT - QUICK SETUP SCRIPT (Linux/macOS)
# ============================================================================
# This script sets up the entire project environment
# Run as: bash setup.sh
# ============================================================================

set -e  # Exit on error

echo ""
echo "============================================================================"
echo "   TOMATO DISEASE DETECTION SYSTEM - SETUP WIZARD (Linux/macOS)"
echo "============================================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.8+ from python.org or using your package manager"
    exit 1
fi

echo "[✓] Python found"
python3 --version

echo ""
echo "============================================================================"
echo "Step 1: Checking existing virtual environment..."
echo "============================================================================"
if [ -d "venv" ]; then
    echo "[✓] Virtual environment folder exists"
    read -p "Do you want to use existing venv? (y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        echo "[*] Removing old venv..."
        rm -rf venv
        echo "[✓] Old venv removed"
    else
        . venv/bin/activate
        echo "[✓] Virtual environment activated"
        goto_step_2=1
    fi
fi

if [ "$goto_step_2" != "1" ]; then
    echo "[*] Creating new virtual environment..."
    python3 -m venv venv
    echo "[✓] Virtual environment created"
    
    echo "[*] Activating virtual environment..."
    . venv/bin/activate
    echo "[✓] Virtual environment activated"
fi

echo ""
echo "============================================================================"
echo "Step 2: Upgrading pip, setuptools, and wheel..."
echo "============================================================================"
python -m pip install --upgrade pip setuptools wheel
echo "[✓] Pip upgraded"

echo ""
echo "============================================================================"
echo "Step 3: Installing core dependencies (PyTorch)..."
echo "============================================================================"
echo ""
echo "Choose installation mode:"
echo "1) GPU Support (CUDA 11.8) - Recommended for faster training"
echo "2) GPU Support (CUDA 12.1) - For newer NVIDIA GPUs"
echo "3) CPU Only - Works but slower"
echo ""
read -p "Enter choice (1, 2, or 3): " choice

case $choice in
    1)
        echo "[*] Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "[*] Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    *)
        echo "[*] Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio
        ;;
esac
echo "[✓] PyTorch installed"

echo ""
echo "============================================================================"
echo "Step 4: Installing all project dependencies..."
echo "============================================================================"
echo "[*] Installing from requirements.txt..."
if pip install -r requirements.txt; then
    echo "[✓] All dependencies installed successfully!"
else
    echo "[!] Some packages failed to install, but continuing..."
fi

echo ""
echo "============================================================================"
echo "Step 5: Installing missing critical packages..."
echo "============================================================================"
echo "[*] Installing pytorch-grad-cam..."
pip install pytorch-grad-cam
echo "[*] Installing streamlit-webrtc..."
pip install streamlit-webrtc
echo "[*] Installing scikit-learn..."
pip install scikit-learn
echo "[*] Installing pyserial..."
pip install pyserial
echo "[✓] Critical packages installed"

# For Linux, may need system dependencies
if command -v apt-get &> /dev/null; then
    echo ""
    echo "[*] Detected Debian/Ubuntu system"
    read -p "Install system dependencies for Streamlit WebRTC? (y/n): " install_sys_deps
    if [ "$install_sys_deps" = "y" ]; then
        echo "[*] Installing system packages..."
        sudo apt-get update
        sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev libopus-dev libvpx-dev
        echo "[✓] System dependencies installed"
    fi
fi

echo ""
echo "============================================================================"
echo "Step 6: Verifying installation..."
echo "============================================================================"
echo ""
echo "Checking core packages..."

python -c "import torch; print('✓ PyTorch', torch.__version__)" 2>/dev/null && echo "[✓] PyTorch" || echo "[!] PyTorch failed"
python -c "import cv2; print('✓ OpenCV', cv2.__version__)" 2>/dev/null && echo "[✓] OpenCV" || echo "[!] OpenCV failed"
python -c "import numpy; print('✓ NumPy')" 2>/dev/null && echo "[✓] NumPy" || echo "[!] NumPy failed"
python -c "from ultralytics import YOLO; print('✓ YOLO loaded')" 2>/dev/null && echo "[✓] YOLO" || echo "[!] YOLO failed"
python -c "from pytorch_grad_cam import GradCAM; print('✓ Grad-CAM loaded')" 2>/dev/null && echo "[✓] Grad-CAM" || echo "[!] Grad-CAM failed"
python -c "import streamlit; print('✓ Streamlit loaded')" 2>/dev/null && echo "[✓] Streamlit" || echo "[!] Streamlit failed"
python -c "import streamlit_webrtc; print('✓ WebRTC loaded')" 2>/dev/null && echo "[✓] Streamlit WebRTC" || echo "[!] Streamlit WebRTC failed"

echo ""
echo "============================================================================"
echo "Step 7: Checking model files..."
echo "============================================================================"
if [ -f "tomato_disease_model.pth" ]; then
    echo "[✓] tomato_disease_model.pth found"
else
    echo "[!] tomato_disease_model.pth NOT FOUND - Place in project root"
fi

if [ -f "classes.pth" ]; then
    echo "[✓] classes.pth found"
else
    echo "[!] classes.pth NOT FOUND - Place in project root"
fi

if [ -f "best_leaf_detector.pt" ]; then
    echo "[✓] best_leaf_detector.pt found"
else
    echo "[!] best_leaf_detector.pt NOT FOUND - Place in project root"
fi

echo ""
echo "============================================================================"
echo "SETUP COMPLETE!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify all model files are in the project root directory"
echo "  2. Run the Streamlit app: streamlit run app.py"
echo "  3. Or test with CLI: python tomato.py --image test_image.jpg"
echo ""
echo "To activate venv in future sessions:"
echo "  . venv/bin/activate"
echo ""
