import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import sys

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ===============================
# CONFIG
# ===============================

YOLO_MODEL = "best_leaf_detector.pt"
CLASSIFIER_MODEL = "tomato_disease_model.pth"
CLASSES_PATH = "classes.pth"

CONF_THRESHOLD = 0.60  # reject weak predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# LOAD MODELS
# ===============================

leaf_detector = YOLO(YOLO_MODEL)

classes = torch.load(CLASSES_PATH, weights_only=False)
num_classes = len(classes)

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT  # ✅ FIXED
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    torch.load(CLASSIFIER_MODEL, map_location=device)
)

model = model.to(device)
model.eval()


# ===============================
# TRANSFORM
# ===============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ===============================
# DISEASE DATABASE
# ===============================

disease_db = {
    "Bacterial_spot": ("Avoid wet foliage.", "Use copper fungicides."),
    "Early_blight": ("Rotate crops.", "Use Mancozeb."),
    "Late_blight": ("Destroy infected plants.", "Use Dimethomorph."),
    "Leaf_Mold": ("Reduce humidity.", "Use Copper sprays."),
    "Septoria_leaf_spot": ("Use mulch.", "Apply Chlorothalonil."),
    "Spider_mites Two-spotted_spider_mite": ("Reduce stress.", "Use Neem oil."),
    "Target_Spot": ("Keep leaves dry.", "Use fungicides."),
    "Tomato_Yellow_Leaf_Curl_Virus": ("Control whiteflies.", "Remove infected plants."),
    "Tomato_mosaic_virus": ("Use clean seeds.", "Destroy infected plants."),
    "healthy": ("Maintain care.", "No treatment."),
    "powdery_mildew": ("Improve airflow.", "Use sulfur sprays.")
}


# ===============================
# TRUE SEVERITY USING GRADCAM
# ===============================

def severity_from_cam(cam):
    infected_ratio = np.mean(cam > 0.5)

    if infected_ratio > 0.5:
        return "Severe"
    elif infected_ratio > 0.25:
        return "Moderate"
    else:
        return "Mild"


# ===============================
# PROCESS SINGLE LEAF
# ===============================

def process_leaf(leaf_img):

    leaf_rgb = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB)
    leaf_resized = cv2.resize(leaf_rgb, (224, 224))
    rgb_img = np.float32(leaf_resized) / 255

    pil_image = Image.fromarray(leaf_resized)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Classification
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    conf = confidence.item()

    if conf < CONF_THRESHOLD:
        return None  # ❌ reject weak predictions

    disease = classes[pred.item()]

    # Grad-CAM
    target_layers = [model.features[-2]]  # ✅ improved layer

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    severity = severity_from_cam(grayscale_cam)

    return {
        "disease": disease,
        "confidence": conf,
        "severity": severity,
        "cam_image": cam_image,
        "cam_map": grayscale_cam
    }


# ===============================
# MAIN PIPELINE
# ===============================

def run_pipeline(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found")
        return

    results = leaf_detector(image)

    if len(results[0].boxes) == 0:
        print("❌ No leaves detected")
        return

    all_results = []

    # 🔥 MULTI-LEAF PROCESSING
    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        leaf = image[y1:y2, x1:x2]

        result = process_leaf(leaf)

        if result is None:
            continue

        result["box"] = (x1, y1, x2, y2)
        all_results.append(result)

    if len(all_results) == 0:
        print("⚠ No confident predictions")
        return

    # 🔥 PICK WORST (highest severity + confidence)
    def score(r):
        severity_score = {"Mild": 1, "Moderate": 2, "Severe": 3}
        return severity_score[r["severity"]] + r["confidence"]

    best = max(all_results, key=score)

    disease_key = best["disease"].split("___")[-1]
    prevention, treatment = disease_db.get(
        disease_key,
        ("Consult expert", "Consult expert")
    )

    # ===============================
    # OUTPUT
    # ===============================

    print("\n===== FINAL DIAGNOSIS =====")
    print("Disease   :", best["disease"])
    print("Confidence:", f"{best['confidence']*100:.2f}%")
    print("Severity  :", best["severity"])

    print("\nPrevention:", prevention)
    print("Treatment :", treatment)
    print("===========================")

    # ===============================
    # VISUALIZATION
    # ===============================

    for r in all_results:
        x1, y1, x2, y2 = r["box"]

        color = (0, 0, 255) if r["severity"] == "Severe" else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Detected Leaves", image)
    cv2.imshow("Best Leaf GradCAM",
               cv2.cvtColor(best["cam_image"], cv2.COLOR_RGB2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# ENTRY
# ===============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python final_pipeline.py image.jpg")
        exit()

    run_pipeline(sys.argv[1])