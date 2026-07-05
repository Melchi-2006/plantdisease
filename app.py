import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Try to import Grad-CAM (may not be available on newer Python versions)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    st.warning("⚠️ Grad-CAM visualization not available (requires Python 3.11 or 3.12)")

# Try to import WebRTC (may not be available on newer Python versions)
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Tomato Disease AI", layout="wide")

YOLO_MODEL = "best_leaf_detector.pt"
CLASSIFIER_MODEL = "tomato_disease_model.pth"
CLASSES_PATH = "classes.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main-title {font-size:40px;font-weight:bold;}
.card {background:#161B22;padding:20px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL)

    classes = torch.load(CLASSES_PATH)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location=device))
    model = model.to(device)
    model.eval()

    return yolo, model, classes

yolo, model, classes = load_models()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# DISEASE DB
# =========================
disease_db = {
    "Early_blight": ("Crop rotation", "Apply fungicide"),
    "Late_blight": ("Avoid wet leaves", "Use copper fungicide"),
    "Septoria_leaf_spot": ("Remove infected leaves", "Apply mancozeb"),
    "Leaf_Mold": ("Reduce humidity", "Apply fungicide"),
    "healthy": ("Maintain care", "No treatment needed")
}

# =========================
# SEVERITY FUNCTION
# =========================
def severity(conf, area):
    if conf > 0.9 and area > 40:
        return "Severe"
    elif conf > 0.75:
        return "Moderate"
    else:
        return "Mild"

# =========================
# CAMERA CLASS
# =========================
if WEBRTC_AVAILABLE:
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame = img
            return img

# =========================
# UI HEADER
# =========================
st.markdown('<div class="main-title">🍅 Tomato Disease AI System</div>', unsafe_allow_html=True)

input_options = ["Upload Image"]
if WEBRTC_AVAILABLE:
    input_options.append("Camera")

mode = st.sidebar.radio("Select Input Mode", input_options)

image = None

# =========================
# INPUT
# =========================
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded))

elif mode == "Camera" and WEBRTC_AVAILABLE:
    st.subheader("📷 Camera Feed")

    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

    if ctx.video_processor:
        if st.button("📸 Capture"):
            image = ctx.video_processor.frame

# =========================
# MAIN PIPELINE
# =========================
if image is not None:

    st.image(image, caption="Input Image", use_container_width=True)

    results = yolo(image)

    if len(results[0].boxes) == 0:
        st.error("No leaves detected")
        st.stop()

    boxes = results[0].boxes
    img_area = image.shape[0] * image.shape[1]

    leaf_results = []

    for i, box in enumerate(boxes.xyxy):

        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        conf_det = boxes.conf[i].item()

        box_area = (x2 - x1) * (y2 - y1)

        # 🔥 FILTER BAD BOXES
        if box_area > img_area * 0.8:
            continue
        if box_area < 2000:
            continue

        # padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        leaf = image[y1:y2, x1:x2]

        if leaf.shape[0] < 50 or leaf.shape[1] < 50:
            continue

        # =========================
        # CLASSIFICATION
        # =========================
        leaf_rgb = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
        leaf_resized = cv2.resize(leaf_rgb, (224,224))

        pil_img = Image.fromarray(leaf_resized)

        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf_cls, pred = torch.max(probs, 1)

        disease = classes[pred.item()]
        confidence = conf_cls.item()

        # =========================
        # GRADCAM
        # =========================
        cam_image = None
        infection_percent = 0
        
        if GRADCAM_AVAILABLE:
            cam = GradCAM(model=model, target_layers=[model.features[-1]])

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(pred.item())]
            )[0]

            rgb_img = np.float32(leaf_resized) / 255
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # =========================
            # INFECTION %
            # =========================
            infected = (grayscale_cam > 0.5).sum() / grayscale_cam.size
            infection_percent = round(infected * 100, 2)
        else:
            # Use simple visualization without Grad-CAM
            cam_image = leaf_resized

        leaf_results.append({
            "box": (x1, y1, x2, y2),
            "leaf_img": leaf,
            "cam": cam_image,
            "disease": disease,
            "confidence": confidence,
            "infection": infection_percent
        })

    # =========================
    # DRAW ALL BOXES
    # =========================
    annotated = image.copy()

    for res in leaf_results:
        x1,y1,x2,y2 = res["box"]
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

    st.image(annotated, caption="Detected Leaves", use_container_width=True)

    # =========================
    # DISPLAY EACH LEAF
    # =========================
    st.subheader("🌿 Leaf Analysis")

    for i, res in enumerate(leaf_results):

        st.markdown(f"### Leaf {i+1}")

        c1, c2 = st.columns(2)

        with c1:
            st.image(res["leaf_img"], caption="Leaf Crop")

        with c2:
            if GRADCAM_AVAILABLE:
                st.image(res["cam"], caption="Grad-CAM Heatmap")
            else:
                st.image(res["cam"], caption="Leaf Image (Grad-CAM unavailable)")

        sev = severity(res["confidence"], res["infection"])

        st.markdown(f"""
        **Disease:** {res["disease"]}  
        **Confidence:** {res["confidence"]*100:.2f}%  
        **Severity:** {sev}  
        **Infection Area:** {res["infection"]}%  
        """)

    # =========================
    # FINAL SUMMARY
    # =========================
    st.subheader("📊 Final Diagnosis")

    if len(leaf_results) > 0:

        best_leaf = max(
            leaf_results,
            key=lambda x: (x["infection"], x["confidence"])
        )

        disease_key = best_leaf["disease"].split("___")[-1]

        prevention, treatment = disease_db.get(
            disease_key,
            ("Consult expert","Consult expert")
        )

        st.success(f"""
        🌱 Most Affected Disease: {best_leaf['disease']}

        Confidence: {best_leaf['confidence']*100:.2f}%  
        Infection Area: {best_leaf['infection']}%
        """)

        st.markdown(f"""
        <div class="card">
        <h3>Prevention</h3>
        <p>{prevention}</p>
        <h3>Treatment</h3>
        <p>{treatment}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("No valid leaves detected")