# 🍅 TOMATO DISEASE DETECTION PROJECT - COMPLETE OVERVIEW

---

## 🎯 PROJECT SUMMARY

This is a **state-of-the-art AI system** for automated tomato disease detection, combining:

- **Object Detection (YOLO)** - Identifies tomato leaves
- **Disease Classification (EfficientNet-B0)** - Classifies 11 disease types
- **Model Explainability (Grad-CAM)** - Shows affected regions
- **Web Interface (Streamlit)** - User-friendly UI
- **Real-time Processing** - Live video analysis
- **Edge Deployment** - Raspberry Pi support

**Supported Diseases**: 11 types + Healthy classification

---

## 🚀 INSTANT START

### Windows Users:
```bash
setup.bat
```

### Linux/macOS Users:
```bash
chmod +x setup.sh
./setup.sh
```

This **one command** will:
1. Create virtual environment
2. Install all dependencies
3. Verify installation
4. Check model files

---

## 📋 WHAT THIS PROJECT INCLUDES

### Main Scripts (Ready to Use)
- **`app.py`** - Interactive web interface (⭐ START HERE)
- **`tomato.py`** - Command-line prediction
- **`leaf.py`** - Grad-CAM visualization
- **`train_gpu.py`** - Model training
- **`tomcla.py`** - Advanced CLI with logging
- **`tompi.py`** - Raspberry Pi optimized

### Dataset
- **Training**: 11 disease classes + healthy
- **Validation**: Separate validation set
- **Test**: Pre-labeled test images
- **Format**: Standard image folders

### Pre-trained Models
- **`tomato_disease_model.pth`** - EfficientNet classifier (100+ MB)
- **`classes.pth`** - Disease labels
- **`best_leaf_detector.pt`** - YOLOv8 object detector

### Documentation (Created for You)
- **`PROJECT_DOCUMENTATION.md`** - Comprehensive guide
- **`QUICK_START.md`** - Quick reference
- **`ENVIRONMENT_STATUS.md`** - Dependencies status
- **`requirements.txt`** - Python packages
- **`setup.bat / setup.sh`** - Automated setup

---

## 🔧 INSTALLATION (3 Options)

### Option 1: Automatic (Recommended) ⭐
```bash
# Windows
setup.bat

# Linux/macOS  
./setup.sh
```

### Option 2: Manual
```bash
python -m venv venv
venv\Scripts\activate  # Windows: . venv/bin/activate (Linux)
pip install -r requirements.txt
```

### Option 3: Using Conda
```bash
conda create -n tomato python=3.11
conda activate tomato
pip install -r requirements.txt
```

---

## ⚠️ MISSING PACKAGES (4 Total)

These must be installed (auto-installed by setup.bat/setup.sh):

```bash
pip install pytorch-grad-cam streamlit-webrtc scikit-learn pyserial
```

| Package | Purpose | Critical? |
|---------|---------|-----------|
| pytorch-grad-cam | Model visualization | ✅ Yes |
| streamlit-webrtc | Webcam in web app | ✅ Yes |
| scikit-learn | Metrics & analysis | ⚠️ For training |
| pyserial | ESP32 communication | ❌ Optional |

---

## 📊 INSTALLED PACKAGES (80+ Total)

### Core ML Stack
- PyTorch 2.12.1
- TorchVision 0.27.1
- NumPy 2.4.4
- OpenCV 5.0.0.93

### Computer Vision
- YOLOv8 (ultralytics 8.4.86)
- Pillow 12.1.1
- Matplotlib 3.11.0

### Web Framework
- Streamlit 1.55.0
- FastAPI components (starlette, uvicorn)

### All 80+ installed packages listed in `ENVIRONMENT_STATUS.md`

---

## 🎯 HOW TO RUN (5 Options)

### 1️⃣ Web Interface (Best for Everyone)
```bash
streamlit run app.py
# Opens http://localhost:8501
# Upload image or use camera
```
**Best for**: Quick testing, user-friendly

### 2️⃣ Command Line (Fastest)
```bash
python tomato.py --image path/to/image.jpg
# Single command, instant output
```
**Best for**: Scripting, batch processing

### 3️⃣ Advanced CLI
```bash
python tomcla.py --camera
# Real-time with logging
```
**Best for**: Monitoring, logging

### 4️⃣ Visualization
```bash
python leaf.py --image image.jpg
# Shows which parts of leaf are diseased
```
**Best for**: Understanding model decisions

### 5️⃣ Training
```bash
python train_gpu.py
# Retrain on your dataset
```
**Best for**: Advanced users, custom models

---

## 📁 PROJECT STRUCTURE

```
tomato_project/
├── 📄 PROJECT_DOCUMENTATION.md   ← Full details
├── 📄 QUICK_START.md             ← Quick reference  
├── 📄 ENVIRONMENT_STATUS.md      ← Dependencies
├── 📄 requirements.txt           ← Package list
├── 🔧 setup.bat / setup.sh       ← Run first!
│
├── 🤖 Models (Required):
│   ├── tomato_disease_model.pth  ← Classifier
│   ├── classes.pth               ← Labels
│   └── best_leaf_detector.pt     ← Detector
│
├── 💻 Main Scripts:
│   ├── app.py                    ← Web UI ⭐
│   ├── tomato.py                 ← CLI
│   ├── leaf.py                   ← Visualization
│   ├── train_gpu.py              ← Training
│   ├── tomcla.py                 ← Advanced
│   └── tompi.py                  ← Raspberry Pi
│
├── 📊 Dataset:
│   ├── train/   (11 disease classes)
│   ├── valid/   (11 disease classes)
│   └── test/    (11 disease classes)
│
└── 🎯 YOLO Leaf Detection:
    └── yololeaf/
        ├── train/
        ├── valid/
        └── test/
```

---

## 🦠 DISEASE CLASSES (11 + Healthy)

1. Bacterial_spot
2. Early_blight
3. Late_blight
4. Leaf_Mold
5. Septoria_leaf_spot
6. Spider_mites Two-spotted_spider_mite
7. Target_Spot
8. Tomato_mosaic_virus
9. Tomato_Yellow_Leaf_Curl_Virus
10. powdery_mildew
11. **healthy**

---

## 💻 SYSTEM REQUIREMENTS

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8+ GB |
| Disk | 2 GB | 5+ GB |
| GPU | None | NVIDIA 4GB+ VRAM |
| OS | Windows 10+ | Windows 11 / Ubuntu 20.04+ |

---

## 🎓 MODEL ARCHITECTURE

### Leaf Detection (YOLO)
- Model: YOLOv8 Nano
- Input: Full image (any size)
- Output: Bounding boxes
- Speed: ~100ms (GPU)

### Disease Classification
- Model: EfficientNet-B0
- Input: 224×224 RGB image
- Output: 11 class probabilities
- Accuracy: ~95% on test set

### Explainability
- Method: Grad-CAM
- Shows: Which regions trigger disease detection
- Output: Heatmap overlay

---

## ⚡ PERFORMANCE

| Metric | GPU | CPU |
|--------|-----|-----|
| Single Image | 0.1 sec | 2-5 sec |
| Video Frame | 0.05 sec | 1-2 sec |
| Memory | ~1.5 GB | ~500 MB |
| Model Size | 150 MB | 150 MB |

---

## 🐛 TROUBLESHOOTING QUICK FIXES

| Problem | Solution |
|---------|----------|
| "Module not found" | `pip install pytorch-grad-cam streamlit-webrtc` |
| No GPU detected | `python -c "import torch; print(torch.cuda.is_available())"` |
| Slow performance | Use GPU or Raspberry Pi version |
| App won't start | Reinstall streamlit-webrtc |
| Model not found | Verify model files in project root |

---

## 📞 KEY COMMANDS

```bash
# Setup
setup.bat                          # Windows automatic setup
./setup.sh                         # Linux/macOS automatic setup

# Run
streamlit run app.py               # Web interface
python tomato.py --image img.jpg   # Single prediction
python tomcla.py --camera          # Live camera feed
python leaf.py --image img.jpg     # Grad-CAM visualization

# Training
python train_gpu.py                # Train model
python clean_dataset.py            # Clean corrupted images

# Verify
pip install pytorch-grad-cam       # Install missing packages
python -c "import torch; ..."      # Test imports
```

---

## 📊 WHAT YOU GET

✅ Production-ready AI system  
✅ 11 disease classifications  
✅ Real-time video processing  
✅ Model explainability (Grad-CAM)  
✅ Web + CLI interfaces  
✅ Pre-trained weights  
✅ Full dataset  
✅ Training scripts  
✅ Raspberry Pi support  

---

## 🎯 NEXT STEPS

1. **Run Setup**
   ```bash
   setup.bat  # or ./setup.sh
   ```

2. **Install Missing Packages**
   ```bash
   pip install pytorch-grad-cam streamlit-webrtc scikit-learn pyserial
   ```

3. **Verify Installation**
   ```bash
   python -c "from pytorch_grad_cam import GradCAM; print('Ready!')"
   ```

4. **Launch Web App**
   ```bash
   streamlit run app.py
   ```

5. **Upload Test Image**
   - Open http://localhost:8501
   - Upload an image
   - See disease prediction

---

## 📚 DOCUMENTATION FILES CREATED

| File | Purpose |
|------|---------|
| `PROJECT_DOCUMENTATION.md` | 📖 Complete guide (150+ lines) |
| `QUICK_START.md` | ⚡ Quick reference |
| `ENVIRONMENT_STATUS.md` | 📊 Dependencies status |
| `requirements.txt` | 📦 Python packages |
| `setup.bat` | 🔧 Windows setup script |
| `setup.sh` | 🔧 Linux/macOS setup script |
| `PROJECT_OVERVIEW.md` | 📋 This file |

---

## 🏁 FINAL CHECKLIST

Before you start:

- [ ] Python 3.8+ installed
- [ ] Project files extracted
- [ ] Model files in root directory:
  - [ ] `tomato_disease_model.pth`
  - [ ] `classes.pth`
  - [ ] `best_leaf_detector.pt`
- [ ] Read this file (you're here!)
- [ ] Run `setup.bat` or `./setup.sh`
- [ ] Install missing packages
- [ ] Run `streamlit run app.py`

---

## 🎓 HOW IT WORKS (Simple Explanation)

1. **Upload Image** → Shows leaf in photo
2. **Detect Leaf** → YOLOv8 finds the leaf
3. **Extract Leaf** → Crop leaf region  
4. **Classify Disease** → EfficientNet predicts disease
5. **Explain Decision** → Grad-CAM shows affected areas
6. **Show Results** → Disease name + confidence + treatment

---

## 📞 SUPPORT RESOURCES

- **Full Docs**: Open `PROJECT_DOCUMENTATION.md`
- **Quick Help**: Open `QUICK_START.md`
- **Dependencies**: Open `ENVIRONMENT_STATUS.md`
- **Install**: Run `setup.bat` or `setup.sh`

---

## ✨ PROJECT HIGHLIGHTS

🎯 **Accuracy**: ~95% on test dataset  
⚡ **Speed**: Real-time processing (GPU)  
🔍 **Interpretability**: Grad-CAM visualization  
🌐 **Web Interface**: Streamlit app  
💻 **CLI Support**: Command-line tools  
🍓 **Edge Ready**: Raspberry Pi version  
📚 **Well Documented**: 5+ guide files  
🔧 **Easy Setup**: Single setup script  

---

## 🚀 YOU'RE READY!

**Status**: ✅ All documentation complete  
**Setup Time**: ~10 minutes  
**Time to First Prediction**: ~2 minutes  
**Start**: Run `setup.bat` (Windows) or `./setup.sh` (Linux/Mac)

**Good luck! 🍅**

---

*Last Updated: July 5, 2026*  
*Python Version: 3.14.3*  
*Status: Ready for Deployment*
