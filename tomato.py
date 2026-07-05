import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import sys

# ===============================
# CONFIG
# ===============================
YOLO_MODEL = "best_leaf_detector.pt"
CLASSIFIER_MODEL = "tomato_disease_model.pth"
CLASSES_PATH = "classes.pth"

CONF_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# LOAD MODELS
# ===============================
leaf_detector = YOLO(YOLO_MODEL)

classes = torch.load(CLASSES_PATH, weights_only=False)
num_classes = len(classes)

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
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
# LEAF MASK (Largest Contour)
# ===============================
def get_leaf_mask(leaf_img):

    hsv = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return mask

    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    return clean_mask

# ===============================
# SMART SEVERITY
# ===============================
def estimate_severity(leaf_img):

    leaf_img = cv2.convertScaleAbs(leaf_img, alpha=1.3, beta=15)
    hsv = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)

    leaf_mask = get_leaf_mask(leaf_img)

    # Disease detection
    lower_yellow = np.array([18, 40, 80])
    upper_yellow = np.array([45, 255, 255])

    lower_pale = np.array([20, 20, 140])
    upper_pale = np.array([50, 120, 255])

    lower_brown = np.array([5, 70, 20])
    upper_brown = np.array([25, 255, 180])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_pale = cv2.inRange(hsv, lower_pale, upper_pale)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    disease_mask = mask_yellow | mask_pale | mask_brown

    # Clean noise
    kernel = np.ones((3,3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5,5), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)

    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    infected_pixels = np.sum(disease_mask > 0)
    leaf_pixels = np.sum(leaf_mask > 0)

    ratio = infected_pixels / (leaf_pixels + 1e-6)

    # Ignore tiny noise
    if ratio < 0.02:
        return "Mild", 0.0

    if ratio > 0.25:
        return "Severe", ratio
    elif ratio > 0.12:
        return "Moderate", ratio
    else:
        return "Mild", ratio


# ===============================
# PROCESS LEAF
# ===============================
def process_leaf(leaf_img):

    leaf_rgb = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB)
    leaf_resized = cv2.resize(leaf_rgb, (224, 224))

    pil_image = Image.fromarray(leaf_resized)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    conf = confidence.item()
    disease = classes[pred.item()]

    if conf < CONF_THRESHOLD:
        return None

    # ✅ FIX: Healthy leaf handling
    if "healthy" in disease.lower():
        return {
            "disease": disease,
            "confidence": conf,
            "severity": "None",
            "damage_ratio": 0.0
        }

    severity, ratio = estimate_severity(leaf_img)

    return {
        "disease": disease,
        "confidence": conf,
        "severity": severity,
        "damage_ratio": ratio
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

    # Pick best result
    def score(r):
        severity_score = {"None":0, "Mild":1, "Moderate":2, "Severe":3}
        return severity_score[r["severity"]] + r["confidence"]

    best = max(all_results, key=score)

    disease_key = best["disease"].split("___")[-1]
    prevention, treatment = disease_db.get(
        disease_key,
        ("Consult expert", "Consult expert")
    )

    # ================= OUTPUT =================
    print("\n===== FINAL DIAGNOSIS =====")
    print("Disease   :", best["disease"])
    print("Confidence:", f"{best['confidence']*100:.2f}%")
    print("Severity  :", best["severity"])
    print("Affected  :", f"{best['damage_ratio']*100:.2f}%")

    print("\nPrevention:", prevention)
    print("Treatment :", treatment)
    print("===========================")

    # ================= VISUAL =================
    for r in all_results:
        x1, y1, x2, y2 = r["box"]

        if r["severity"] == "Severe":
            color = (0, 0, 255)
        elif r["severity"] == "Moderate":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Detected Leaves", image)
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