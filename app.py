import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
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
YOLO_MODEL = BASE_DIR / "best_leaf_detector.pt"

DISEASE_DB = {
    "Bacterial_spot": (
        "Avoid working in fields when foliage is wet. Use certified seed.",
        "Apply copper fungicides mixed with Mancozeb to overcome copper-resistant bacteria.",
    ),
    "Early_blight": (
        "Rotate crops for 2+ years and use disease-free seeds.",
        "Prune infected lower branches and spray with Mancozeb or copper fungicides weekly.",
    ),
    "Late_blight": (
        "Destroy volunteer potato/tomato plants nearby and use morning sun exposure to dry dew.",
        "Apply Dimethomorph (Acrobat) or copper fungicides immediately at first sight.",
    ),
    "Leaf_Mold": (
        "Keep relative humidity below 85% in greenhouses and prune for air circulation.",
        "Use fungicides containing Copper or Thiophanate Methyl if outbreaks occur in high humidity.",
    ),
    "Septoria_leaf_spot": (
        "Mulch heavily to prevent spores in the soil from splashing onto leaves.",
        "Remove infected lower leaves and apply Chlorothalonil or copper-based sprays every 7–10 days.",
    ),
    "Spider_mites Two-spotted_spider_mite": (
        "Avoid excessive nitrogen fertilizer and keep plants well-hydrated.",
        "Spray undersides of leaves with Neem Oil, insecticidal soap, or targeted miticides.",
    ),
    "Target_Spot": (
        "Keep foliage dry and prune to improve sunlight penetration.",
        "Use broad-spectrum fungicides and avoid working in the garden when leaves are wet.",
    ),
    "Tomato_Yellow_Leaf_Curl_Virus": (
        "Control whiteflies using yellow sticky traps and reflective mulches.",
        "Remove infected plants in bags to trap whiteflies and use organic viricides.",
    ),
    "Tomato_mosaic_virus": (
        "Use certified virus-free seeds and avoid handling plants if you use tobacco.",
        "Pull out and burn infected plants immediately; do not compost them.",
    ),
    "healthy": (
        "Maintain balanced irrigation and fertilization.",
        "No treatment needed.",
    ),
    "powdery_mildew": (
        "Space plants 18–24 inches apart for maximum airflow.",
        "Apply sulfur-based sprays, potassium bicarbonate, or systemic fungicides.",
    ),
}


def severity_level(confidence: float) -> str:
    if confidence > 0.90:
        return "Severe"
    elif confidence > 0.75:
        return "Moderate"
    elif confidence > 0.60:
        return "Mild"
    return "Uncertain"


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

    yolo = None
    gradcam_tools = None
    try:
        from ultralytics import YOLO

        if YOLO_MODEL.exists():
            yolo = YOLO(str(YOLO_MODEL))
    except Exception:
        yolo = None

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        gradcam_tools = {
            "GradCAM": GradCAM,
            "show_cam_on_image": show_cam_on_image,
            "ClassifierOutputTarget": ClassifierOutputTarget,
        }
    except Exception:
        gradcam_tools = None

    return model, classes, transform, device, yolo, gradcam_tools


def detect_leaves(image: Image.Image, yolo_detector):
    if yolo_detector is None:
        return []

    image_np = np.array(image)
    results = yolo_detector(image_np)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return []

    boxes = []
    for box in results[0].boxes.xyxy:
        boxes.append(tuple(map(int, box.cpu().numpy())))
    return boxes


def draw_annotations(image: Image.Image, boxes, labels=None):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        label = labels[idx - 1] if labels is not None and idx - 1 < len(labels) else f"Leaf {idx}"
        draw.text((x1 + 6, y1 + 6), label, fill="white")

    return annotated


def compute_gradcam(pil_image, model, transform, device, gradcam_tools, target_class):
    if gradcam_tools is None:
        return None, None

    pil_image_resized = pil_image.resize((224, 224))
    np_image = np.array(pil_image_resized).astype(np.float32) / 255.0
    input_tensor = transform(pil_image_resized).unsqueeze(0).to(device)

    cam = gradcam_tools["GradCAM"](
        model=model,
        target_layers=[model.features[-1]],
    )
    targets = [gradcam_tools["ClassifierOutputTarget"](target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = gradcam_tools["show_cam_on_image"](np_image, grayscale_cam, use_rgb=True)

    return Image.fromarray(cam_image), grayscale_cam


def analyze_leaf(crop_image, model, classes, transform, device, gradcam_tools):
    tensor = transform(crop_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    disease = classes[pred.item()]
    conf = confidence.item()
    severity = severity_level(conf)
    cam_image, cam_map = compute_gradcam(crop_image, model, transform, device, gradcam_tools, pred.item())

    infection_area = None
    if cam_map is not None:
        infection_area = round(float(np.mean(cam_map > 0.5)) * 100, 2)

    return {
        "disease": disease,
        "confidence": round(conf * 100, 2),
        "severity": severity,
        "cam_image": cam_image,
        "infection_area": infection_area,
        "leaf_crop": crop_image,
    }


def format_disease_key(disease_name: str) -> str:
    return disease_name.split("___")[-1]


st.title("🍅 Tomato Disease Detector")
st.markdown(
    "Upload a tomato leaf image and get a disease diagnosis with annotated leaf regions, Grad-CAM highlights, and treatment guidance."
)

try:
    model, classes, transform, device, yolo_detector, gradcam_tools = load_models()
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

if yolo_detector is None:
    st.warning("Leaf detector is unavailable. The app will still classify the image, but bounding box annotations will not be shown.")

if gradcam_tools is None:
    st.warning("Grad-CAM is unavailable. The app will still classify the image, but heatmap overlays will not be shown.")

uploaded_file = st.file_uploader("Choose a tomato leaf image to analyze", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Leaf"):
        with st.spinner("Detecting and analyzing leaf regions..."):
            boxes = detect_leaves(image, yolo_detector)
            labels = [f"Leaf {i + 1}" for i in range(len(boxes))]
            annotated_image = draw_annotations(image, boxes, labels) if boxes else image

            leaf_results = []
            if boxes:
                for box in boxes:
                    crop = image.crop(box)
                    leaf_results.append(analyze_leaf(crop, model, classes, transform, device, gradcam_tools))
            else:
                leaf_results.append(analyze_leaf(image, model, classes, transform, device, gradcam_tools))

            severity_rank = {"Uncertain": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
            best_leaf = max(
                leaf_results,
                key=lambda x: (severity_rank.get(x["severity"], 0), x["confidence"]),
            )
            disease_key = format_disease_key(best_leaf["disease"])
            prevention, treatment = DISEASE_DB.get(
                disease_key,
                ("Consult a local agricultural expert.", "Consult a local agricultural expert."),
            )

        st.subheader("Annotated Leaf Detection")
        st.image(annotated_image, caption="Detected leaf regions", use_column_width=True)

        if boxes:
            for index, leaf_data in enumerate(leaf_results, start=1):
                st.write(f"### Leaf {index}")
                col1, col2 = st.columns(2)
                col1.image(leaf_data["leaf_crop"], caption="Leaf crop", use_column_width=True)
                if leaf_data["cam_image"] is not None:
                    col2.image(leaf_data["cam_image"], caption="Grad-CAM heatmap", use_column_width=True)
                else:
                    col2.info("Grad-CAM heatmap not available.")

                st.markdown(
                    f"**Disease:** {leaf_data['disease']}\n"
                    f"**Confidence:** {leaf_data['confidence']}%\n"
                    f"**Severity:** {leaf_data['severity']}"
                )
                if leaf_data["infection_area"] is not None:
                    st.markdown(f"**Infection area:** {leaf_data['infection_area']}%")

        else:
            st.write("No leaf boxes were found. Showing the best available prediction for the full image.")
            leaf_data = leaf_results[0]
            st.image(leaf_data["leaf_crop"], caption="Leaf crop", use_column_width=True)
            st.markdown(
                f"**Disease:** {leaf_data['disease']}\n"
                f"**Confidence:** {leaf_data['confidence']}%\n"
                f"**Severity:** {leaf_data['severity']}"
            )
            if leaf_data["infection_area"] is not None:
                st.markdown(f"**Infection area:** {leaf_data['infection_area']}%")
            if leaf_data["cam_image"] is not None:
                st.image(leaf_data["cam_image"], caption="Grad-CAM heatmap", use_column_width=True)

        st.subheader("Recommended Cure and Prevention")
        st.markdown(f"**Disease detected:** {best_leaf['disease']}")
        st.markdown(f"**Best guess confidence:** {best_leaf['confidence']}%")
        st.markdown(f"**Severity level:** {best_leaf['severity']}")
        st.markdown(f"**Prevention:** {prevention}")
        st.markdown(f"**Treatment:** {treatment}")
