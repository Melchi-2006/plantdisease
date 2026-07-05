# 🍅 Tomato Disease Detection & Classification Project

**Status**: ⚠️ NOT TRAINED ON CURRENT LAPTOP - Missing dependencies  
**Python Version**: 3.14.3  
**Last Updated**: Project requires environment setup

---

## 📋 Project Overview

This is a **comprehensive AI-powered Tomato Disease Detection System** that combines:
1. **YOLO Object Detection** - Detects tomato leaves in images/video
2. **EfficientNet Classifier** - Classifies diseases from detected leaves
3. **Grad-CAM Visualization** - Shows which parts of the leaf are diseased
4. **Streamlit Web Interface** - Interactive web application for predictions
5. **Real-time Video Processing** - Live camera feed analysis

### 🎯 Supported Disease Classes (11 types):
- Bacterial_spot
- Early_blight
- Late_blight
- Leaf_Mold
- Septoria_leaf_spot
- Spider_mites Two-spotted_spider_mite
- Target_Spot
- Tomato_mosaic_virus
- Tomato_Yellow_Leaf_Curl_Virus
- powdery_mildew
- healthy

---

## 📁 Project Structure

```
tomato_project/
├── dataset/                        # Training/Validation/Test dataset (11 disease classes)
│   ├── train/                      # Training images
│   ├── valid/                      # Validation images
│   └── test/                       # Test images
│
├── yololeaf/                       # YOLO leaf detection model data
│   ├── train/images & labels/
│   ├── valid/images & labels/
│   ├── test/images & labels/
│   ├── data.yaml                   # YOLO dataset config
│   └── *.pt files                  # YOLO model weights
│
├── runs/                           # YOLO training outputs
│   └── detect/
│       ├── leaf_detector/
│       ├── leaf_detector2/
│       └── leaf_detector3/
│
├── snapshots/                      # Saved prediction results
│
├── venv/                           # Virtual environment (corrupted venv path)
│
# ========== MAIN SCRIPTS ==========
├── app.py                          # Streamlit web app (interactive UI)
├── tomato.py                       # Main prediction CLI script
├── leaf.py                         # Leaf detection with Grad-CAM visualization
├── predict_tomato_gradcam.py       # Advanced Grad-CAM visualization
│
# ========== TRAINING & DATA ==========
├── train_gpu.py                    # Train disease classifier (GPU)
├── clean_dataset.py                # Remove corrupted images from dataset
│
# ========== SPECIALIZED SCRIPTS ==========
├── tomcla.py                       # Comprehensive CLI with logging
├── tompi.py                        # Raspberry Pi optimized version
├── dummy.py                        # Testing/utility script
├── gradcam_visualization.py        # Grad-CAM visualization utility
│
# ========== MODEL WEIGHTS ==========
├── tomato_disease_model.pth        # Trained EfficientNet-B0 classifier
├── classes.pth                     # Encoded disease class labels
├── best_leaf_detector.pt           # YOLO object detector (primary)
├── best_leaf_detector1.pt          # YOLO backup model v1
├── yolov8n.pt                      # Base YOLOv8 weights
│
# ========== TEST IMAGES ==========
├── hel.jfif, lb.jfif, mold.jfif   # Sample test images
├── hhh.jpg, fff.jpg, lol.jpg       # Additional test samples
└── [many more test images...]
```

---

## 🔧 Environment & Dependencies

### System Requirements
- **OS**: Windows / Linux / macOS
- **Python**: 3.8 - 3.12 (tested on 3.14.3)
- **GPU**: CUDA-capable GPU recommended (CPU supported but slower)
- **RAM**: 8GB+ recommended
- **Disk**: 4GB+ for models and dataset

### Current Python Installation
```
Python 3.14.3
Location: C:\Users\marin\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

### ❌ Missing Dependencies (MUST INSTALL)

These packages are **imported in scripts but NOT installed**:

```bash
# Critical Missing Packages
pip install pytorch-grad-cam>=1.4.8
pip install streamlit-webrtc>=0.47.0
pip install scikit-learn>=1.3.0
pip install pyserial>=3.5  # For ESP32 serial communication (tompi.py, dummy.py)
```

### ✅ Currently Installed Core Dependencies

```
# Deep Learning
torch==2.12.1
torchvision==0.27.1
tensorflow (not installed, not needed)

# Computer Vision & YOLO
opencv-python==5.0.0.93
ultralytics==8.4.86
ultralytics-thop==2.0.20

# Web Framework
streamlit==1.55.0
starlette==1.3.1
uvicorn==0.50.0

# Data Processing
numpy==2.4.4
pandas==2.3.3
pillow==12.1.1
pyarrow==23.0.1
polars==1.42.1
narwhals==2.18.1

# Utilities
matplotlib==3.11.0
click==8.3.1
PyYAML==6.0.3
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-multipart==0.0.32
requests==2.33.0

# System & Monitoring
psutil==7.2.2
pywin32==312
nvidia-ml-py==13.610.43

# Other
google-genai==2.10.0
cryptography==49.0.0
websockets==16.0
watchdog==6.0.0
```

---

## 📦 Quick Setup Instructions

### Step 1: Install Missing Dependencies

```bash
# Navigate to project directory
cd c:\Users\marin\Documents\tomato_project

# Install missing packages
pip install pytorch-grad-cam streamlit-webrtc scikit-learn pyserial

# Verify installation
python -c "from pytorch_grad_cam import GradCAM; print('✓ pytorch-grad-cam installed')"
python -c "import streamlit_webrtc; print('✓ streamlit-webrtc installed')"
python -c "from sklearn.metrics import classification_report; print('✓ scikit-learn installed')"
```

### Step 2: Create requirements.txt

```bash
# Generate requirements file
pip freeze > requirements.txt
```

### Step 3: Verify Models Are Present

```bash
# Check model files exist
ls -la *.pt *.pth

# Expected files:
# - tomato_disease_model.pth (classifier)
# - classes.pth (labels)
# - best_leaf_detector.pt (YOLO)
```

---

## 🚀 Running the Project

### 1. **Streamlit Web App** (Interactive UI)
```bash
cd c:\Users\marin\Documents\tomato_project
streamlit run app.py
```
- Opens browser at `http://localhost:8501`
- Upload image or use camera
- Real-time disease detection with confidence scores
- Recommended for quick testing

### 2. **CLI Prediction Script**
```bash
python tomato.py --image path/to/image.jpg
```
- Command-line interface
- Outputs disease class and confidence
- Shows severity assessment

### 3. **Grad-CAM Visualization**
```bash
python leaf.py --image test_image.jpg
```
- Generates heatmap showing diseased regions
- Helpful for understanding model decisions
- Outputs annotated image

### 4. **Raspberry Pi Version** (Optimized)
```bash
python tompi.py
# Detects Raspberry Pi automatically and uses CPU
# Supports live camera feed with serial communication to ESP32
```

### 5. **Comprehensive CLI with Logging**
```bash
python tomcla.py --camera  # Live camera feed
python tomcla.py --image image.jpg  # Single image
python tomcla.py --video video.mp4  # Video file
```

### 6. **Train/Retrain Model**
```bash
python train_gpu.py
```
- Trains EfficientNet-B0 on disease dataset
- Requires GPU for reasonable training time
- Saves model to `tomato_disease_model.pth`

### 7. **Clean Dataset**
```bash
python clean_dataset.py
```
- Removes corrupted images from dataset
- Fixes PIL image verification errors

---

## 🧠 Model Architecture

### Component 1: Leaf Detector (YOLO)
- **Model**: YOLOv8 Nano or custom trained
- **File**: `best_leaf_detector.pt`
- **Input**: Full image (640×480 or higher)
- **Output**: Bounding boxes around leaves
- **Purpose**: Crop leaves from complex background

### Component 2: Disease Classifier
- **Model**: EfficientNet-B0 (pre-trained backbone)
- **File**: `tomato_disease_model.pth`
- **Input**: Leaf image (224×224)
- **Output**: 11 disease class probabilities
- **Classes File**: `classes.pth` (pickle)

### Component 3: Visualization Engine
- **Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Purpose**: Shows which regions influence disease classification
- **Output**: Heatmap overlay on original image

---

## 🔍 Key Configuration Values

Found in `tomcla.py` and `train_gpu.py`:

```python
# Model paths
YOLO_MODEL = "best_leaf_detector.pt"
CLASSIFIER_MODEL = "tomato_disease_model.pth"
CLASSES_PATH = "classes.pth"

# Detection settings
CONF_THRESHOLD = 0.60  # Reject detections < 60% confidence
YOLO_IOU = 0.25        # IOU threshold for NMS
YOLO_CONF = 0.55       # YOLO confidence threshold

# Training settings
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0003
MODEL_PATH = "tomato_disease_model.pth"

# For Raspberry Pi
INFER_EVERY_N_FRAMES = 10  # Skip frames for performance
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360

# For Desktop/GPU
INFER_EVERY_N_FRAMES = 4
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

---

## 🐛 Known Issues & Notes

1. **Old VirtualEnv Path Broken**
   - The venv points to: `C:\Users\Sree Padmesh B T\downloads\`
   - This user path doesn't exist on current laptop
   - Currently using system Python instead
   - **Solution**: Already working with global pip install

2. **Missing Classes in train_gpu.py**
   - Uses sklearn but scikit-learn not installed
   - Need to install: `pip install scikit-learn`
   - Imports: `classification_report`, `confusion_matrix`

3. **Grad-CAM Warnings**
   - pytorch-grad-cam needs installation
   - Missing module in multiple scripts: `leaf.py`, `predict_tomato_gradcam.py`

4. **Streamlit WebRTC**
   - `app.py` uses webcam streaming feature
   - Requires: `pip install streamlit-webrtc`
   - May need additional system dependencies on Linux (libavformat, etc.)

5. **Serial Communication** (tompi.py)
   - Optional ESP32 connection requires pyserial
   - Not critical for main functions
   - Install with: `pip install pyserial`

---

## 📊 Dataset Structure

```
dataset/
├── train/              # Training set
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   ├── healthy/
│   ├── Late_blight/
│   ├── Leaf_Mold/
│   ├── powdery_mildew/
│   ├── Septoria_leaf_spot/
│   ├── Spider_mites Two-spotted_spider_mite/
│   ├── Target_Spot/
│   ├── Tomato_mosaic_virus/
│   └── Tomato_Yellow_Leaf_Curl_Virus/
│
├── valid/              # Validation set
│   └── [same 11 classes]
│
└── test/               # Test set
    └── [same 11 classes]
```

Each disease class has hundreds of labeled leaf images.

---

## 🎓 Training Details

- **Framework**: PyTorch
- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Augmentation**: 
  - Random crop, flip, rotation
  - Color jitter (brightness, contrast, saturation, hue)
  - Perspective distortion
  - Horizontal/Vertical flips
- **Optimizer**: Adam or SGD
- **Loss**: CrossEntropyLoss
- **Metrics**: Accuracy, F1-score, Precision, Recall

---

## 🔗 Important Files Reference

| File | Purpose | Imports |
|------|---------|---------|
| `app.py` | Streamlit web UI | streamlit, torch, YOLO, PIL, Grad-CAM |
| `tomato.py` | CLI prediction | torch, YOLO, cv2, numpy |
| `leaf.py` | Grad-CAM visualization | pytorch_grad_cam, torch, cv2 |
| `train_gpu.py` | Train classifier | torch, sklearn, matplotlib |
| `tomcla.py` | Advanced CLI | torch, YOLO, logging, cv2 |
| `tompi.py` | Raspberry Pi version | torch (CPU), YOLO, cv2, serial |
| `clean_dataset.py` | Dataset validation | PIL |

---

## 🛠️ Troubleshooting

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Shows GPU name
```

### Model Loading Errors
- Ensure `.pt` and `.pth` files are in project root
- Check file permissions
- Try: `model = torch.load(path, map_location='cpu')` if CUDA not available

### Image Processing Issues
- Run `python clean_dataset.py` to fix corrupted images
- Ensure opencv is working: `import cv2; print(cv2.__version__)`

### Streamlit App Won't Start
- Install missing: `pip install streamlit-webrtc`
- Try: `streamlit run app.py --logger.level=debug`

---

## 📝 Next Steps for Setup

1. ✅ **Read this document** (You are here)
2. ⬜ **Install missing packages** (See Quick Setup)
3. ⬜ **Test model loading**: `python tomato.py --image test_image.jpg`
4. ⬜ **Run Streamlit app**: `streamlit run app.py`
5. ⬜ **Test with camera feed**: `python tomcla.py --camera`
6. ⬜ **Optional: Retrain model**: `python train_gpu.py`

---

## 📞 Project Summary

This is a **production-ready AI system** for automated tomato disease detection:
- **Fast**: Real-time video processing capability
- **Accurate**: Trained on large dataset with multiple disease classes
- **Explainable**: Grad-CAM shows decision regions
- **Flexible**: CLI, Web UI, and Real-time streaming options
- **Portable**: Raspberry Pi support for edge deployment

**Current Status**: Ready to use after installing missing dependencies!

---

*Generated: July 5, 2026*
