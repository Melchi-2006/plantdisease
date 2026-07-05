import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms

# Set page config
st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🍅",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_MODEL = BASE_DIR / "tomato_disease_model.pth"
CLASSES_PATH = BASE_DIR / "classes.pth"

# Cache model loading for performance
@st.cache_resource
def load_models():
    if not CLASSIFIER_MODEL.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Model files not found in the project folder")
    
    classes = torch.load(CLASSES_PATH, map_location="cpu")
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    return model, classes, transform, device


def predict(image, model, classes, transform, device):
    """Predict disease from image."""
    image = ImageOps.exif_transpose(image)
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    disease = classes[pred.item()]
    confidence = round(conf.item() * 100, 2)
    
    return disease, confidence


# UI
st.title("🍅 Tomato Disease Detector")
st.markdown("Upload a tomato leaf image to get a disease prediction.")

# Load models
try:
    model, classes, transform, device = load_models()
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    if st.button("Predict Disease"):
        with st.spinner("Processing..."):
            disease, confidence = predict(image, model, classes, transform, device)
        
        st.success(f"**Disease:** {disease}")
        st.info(f"**Confidence:** {confidence}%")