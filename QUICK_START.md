# 🍅 Tomato Disease Detection - QUICK START GUIDE

## ⚡ 30-Second Setup (Windows)

```bash
# 1. Run setup script
setup.bat

# 2. This will automatically:
#    - Create virtual environment
#    - Install all dependencies
#    - Verify installation
#    - Check model files
```

## ⚡ 30-Second Setup (Linux/macOS)

```bash
# 1. Make script executable
chmod +x setup.sh

# 2. Run setup script
./setup.sh

# 3. This will automatically:
#    - Create virtual environment
#    - Install all dependencies
#    - Verify installation
#    - Check model files
```

---

## 📋 Pre-Setup Checklist

Before running setup, ensure you have:

- ✅ Python 3.8+ installed ([Download](https://python.org))
- ✅ Project files extracted to a folder
- ✅ Model files in project root:
  - `tomato_disease_model.pth` (Disease classifier)
  - `classes.pth` (Disease class labels)
  - `best_leaf_detector.pt` (YOLO leaf detector)
- ✅ Dataset in `dataset/` folder (optional for prediction, required for training)

---

## 🚀 Running the Project

### After Setup, Choose One:

### Option 1: Web Interface (Easiest) 🌐
```bash
streamlit run app.py
```
- Opens browser at `http://localhost:8501`
- Upload image or use camera
- Interactive and user-friendly
- **Recommended for beginners**

### Option 2: Command Line (Fastest) 💻
```bash
# Single image
python tomato.py --image path/to/image.jpg

# From camera
python leaf.py --camera

# With Grad-CAM visualization
python leaf.py --image path/to/image.jpg
```

### Option 3: Advanced CLI with Logging 📊
```bash
# Live camera feed
python tomcla.py --camera

# Single image
python tomcla.py --image image.jpg

# Video file
python tomcla.py --video video.mp4

# Real-time with ESP32 serial
python tompi.py
```

### Option 4: Retrain Model (Advanced) 🤖
```bash
# Requires GPU for reasonable speed
python train_gpu.py

# Trains for 20 epochs on dataset/train/
# Saves to: tomato_disease_model.pth
```

### Option 5: Clean Dataset 🧹
```bash
# Remove corrupted images
python clean_dataset.py

# Before retraining if you get image errors
```

---

## 📁 Project Structure Summary

```
tomato_project/
├── setup.bat / setup.sh          ← Run first!
├── requirements.txt              ← Package list
├── PROJECT_DOCUMENTATION.md      ← Full details
│
├── Models (place in root):
│   ├── tomato_disease_model.pth  ← Disease classifier
│   ├── classes.pth               ← Class labels
│   └── best_leaf_detector.pt     ← Leaf detector
│
├── Main Scripts:
│   ├── app.py                    ← Streamlit web UI ⭐
│   ├── tomato.py                 ← CLI prediction
│   ├── leaf.py                   ← Grad-CAM visualization
│   ├── train_gpu.py              ← Train model
│   ├── tomcla.py                 ← Advanced CLI
│   └── tompi.py                  ← Raspberry Pi version
│
├── Dataset:
│   ├── train/                    ← 11 disease classes
│   ├── valid/
│   └── test/
│
└── YOLO Leaf Detection:
    └── yololeaf/
        ├── train/
        ├── valid/
        └── test/
```

---

## ❌ Missing Dependencies (Auto-Installed)

These packages are automatically installed by `setup.bat/setup.sh`:

```
pytorch-grad-cam        # Model explainability (Grad-CAM heatmaps)
streamlit-webrtc        # Webcam streaming for web app
scikit-learn            # ML metrics (confusion matrix, classification report)
pyserial                # ESP32 serial communication (optional)
```

---

## 🔍 Troubleshooting

### "Module not found" Error
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install pytorch-grad-cam streamlit-webrtc scikit-learn pyserial
```

### CUDA/GPU Not Working
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Performance (CPU)
```bash
# Train/inference is much faster on GPU
# Check: python -c "import torch; print(torch.cuda.get_device_name(0))"

# Or use Raspberry Pi optimized version (CPU-optimized)
python tompi.py
```

### Model Files Not Found
```bash
# Verify files exist in project root
ls -la *.pt *.pth  # Linux/macOS
dir *.pt *.pth     # Windows

# Files should be:
# - tomato_disease_model.pth (100+ MB)
# - classes.pth (small, <1 MB)
# - best_leaf_detector.pt (50+ MB)
```

### WebRTC Issues (Camera not working)
```bash
# For Linux, install system packages
sudo apt-get install libavformat-dev libavcodec-dev libavdevice-dev

# Reinstall streamlit-webrtc
pip install --upgrade streamlit-webrtc

# Try using OpenCV instead
python tomato.py --camera
```

---

## 📊 Quick Command Reference

| Task | Command | Time |
|------|---------|------|
| Setup environment | `setup.bat` or `setup.sh` | 5-10 min |
| Launch web app | `streamlit run app.py` | Instant |
| Predict single image | `python tomato.py --image img.jpg` | 2-5 sec |
| Clean dataset | `python clean_dataset.py` | 5-10 min |
| Train model | `python train_gpu.py` | 2-4 hours (GPU) |
| Live camera feed | `python tomcla.py --camera` | Instant |
| Grad-CAM visualization | `python leaf.py --image img.jpg` | 3-5 sec |

---

## 🎯 Recommended Workflow

### First Time Users:
1. Run `setup.bat` (or `setup.sh`)
2. Try web UI: `streamlit run app.py`
3. Upload sample images from dataset
4. Explore Grad-CAM visualizations
5. Test with camera feed

### Advanced Users:
1. Run setup script
2. Use CLI: `python tomato.py --image test.jpg`
3. Integrate with your own code
4. Retrain model with custom dataset

### Production Deployment:
1. Run setup script with GPU support
2. Use Docker for reproducibility
3. Deploy Streamlit app to cloud (Streamlit Cloud, AWS, etc.)
4. Or use CLI in backend service

---

## 📞 Model Performance

- **Detection Speed**: 0.5-2 sec per image (GPU: ~0.1 sec)
- **Accuracy**: ~95%+ on test dataset (varies by class)
- **Memory**: ~1.5 GB (GPU mode), ~500 MB (CPU mode)
- **Classes**: 11 disease types + healthy

---

## 🔐 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| Disk | 2 GB | 5+ GB |
| GPU | None | NVIDIA with 4+ GB VRAM |
| Python | 3.8 | 3.10+ |
| OS | Windows 10+ / Ubuntu 18.04+ | Windows 11 / Ubuntu 20.04+ |

---

## 📝 Environment Variables (Optional)

Create `.env` file for custom settings:

```bash
# .env
DEVICE=cuda  # or cpu
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=0.0003
CONF_THRESHOLD=0.60
```

---

## 🆘 Getting Help

1. **Check PROJECT_DOCUMENTATION.md** for detailed info
2. **Review error messages** - they usually indicate the issue
3. **Run in verbose mode**: `streamlit run app.py --logger.level=debug`
4. **Check logs**: `tail -f tomcla.log` (if using tomcla.py)

---

## ✨ Next Features to Explore

- ✅ Real-time video prediction
- ✅ Batch image processing
- ✅ Model explainability (Grad-CAM)
- ⬜ Custom model retraining
- ⬜ Mobile deployment (TensorFlow Lite)
- ⬜ Raspberry Pi edge deployment
- ⬜ Web API service

---

## 🎓 Learning Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)

---

**Generated**: July 5, 2026  
**Status**: Ready to deploy after setup! ✅

Good luck! 🚀
