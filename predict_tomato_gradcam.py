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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ===============================
# LOAD YOLO LEAF DETECTOR
# ===============================

leaf_detector = YOLO(YOLO_MODEL)


# ===============================
# LOAD CLASSIFIER
# ===============================

classes = torch.load(CLASSES_PATH, weights_only=False)
num_classes = len(classes)

model = models.efficientnet_b0(weights=None)

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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# ===============================
# DISEASE DATABASE
# ===============================

disease_db = {

"Bacterial_spot":(
"Avoid working in fields when foliage is wet. Use certified seed.",
"Apply copper fungicides mixed with Mancozeb to overcome copper-resistant bacteria."
),

"Early_blight":(
"Rotate crops for 2+ years. Use disease-free seeds.",
"Prune infected lower branches. Spray with Mancozeb or copper fungicides weekly."
),

"Late_blight":(
"Destroy volunteer potato/tomato plants nearby. Use morning sun exposure to dry dew.",
"Emergency: Apply Dimethomorph (Acrobat) or copper fungicides immediately at first sight; this disease spreads within hours."
),

"Leaf_Mold":(
"Keep relative humidity below 85% in greenhouses. Prune for air circulation.",
"Use fungicides containing Copper or Thiophanate Methyl if outbreaks occur in high humidity."
),

"Septoria_leaf_spot":(
"Mulch heavily to prevent spores in the soil from splashing onto leaves.",
"Remove infected lower leaves. Apply fungicides like Chlorothalonil or copper-based sprays every 7–10 days."
),

"Spider_mites Two-spotted_spider_mite":(
"Avoid excessive nitrogen fertilizer. Keep plants well-hydrated to reduce stress.",
"Spray undersides of leaves with Neem Oil, insecticidal soap, or targeted miticides."
),

"Target_Spot":(
"Keep foliage dry and prune to improve sunlight penetration.",
"Use broad-spectrum fungicides; avoid working in the garden when leaves are wet."
),

"Tomato_Yellow_Leaf_Curl_Virus":(
"Control Whiteflies (the carriers) using yellow sticky traps and reflective mulches",
"Remove infected plants in bags to trap whiteflies. Boost immunity with organic viricides like Geolife No Virus."
),

"Tomato_mosaic_virus":(
"Use certified virus-free seeds. Avoid handling plants if you use tobacco (can carry the virus).",
"No cure. Pull out and burn infected plants immediately. Do not compost them."
),

"healthy":(
"Maintain balanced irrigation and fertilization.",
"No treatment needed."
),

"powdery_mildew":(
"Space plants 18–24 inches apart for maximum airflow. Choose resistant varieties.",
"Apply sulfur-based sprays, potassium bicarbonate, or systemic fungicides like Thiophanate Methyl (KTM)."
)

}


# ===============================
# SEVERITY ESTIMATION
# ===============================

def severity_level(conf):

    if conf > 0.90:
        return "Severe"

    elif conf > 0.75:
        return "Moderate"

    elif conf > 0.60:
        return "Mild"

    else:
        return "Uncertain"


# ===============================
# PIPELINE FUNCTION
# ===============================

def run_pipeline(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        return


    # ---------------------------------
    # YOLO LEAF DETECTION
    # ---------------------------------

    results = leaf_detector(image)

    if len(results[0].boxes) == 0:
        print("No leaf detected in image")
        return


    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)

    x1,y1,x2,y2 = box

    leaf = image[y1:y2, x1:x2]

    leaf_rgb = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)


    # FIX: resize for GradCAM
    leaf_rgb_resized = cv2.resize(leaf_rgb,(224,224))

    rgb_img = np.float32(leaf_rgb_resized) / 255

    pil_image = Image.fromarray(leaf_rgb_resized)


    # ---------------------------------
    # CLASSIFICATION
    # ---------------------------------

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs,1)


    disease = classes[pred.item()]

    conf = confidence.item()

    severity = severity_level(conf)


    # ---------------------------------
    # GRAD CAM
    # ---------------------------------

    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(pred.item())]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    cam_image = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )


    # ---------------------------------
    # DISEASE INFO
    # ---------------------------------

    disease_key = disease.split("___")[-1]

    prevention, treatment = disease_db.get(
        disease_key,
        ("Consult agricultural expert","Consult agricultural expert")
    )


    # ---------------------------------
    # OUTPUT
    # ---------------------------------

    print("\n========== TOMATO DISEASE DIAGNOSIS ==========")

    print("Disease Detected :", disease)

    print("Confidence       :", f"{conf*100:.2f}%")

    print("Severity Level   :", severity)

    if conf < 0.60:
        print("\n⚠ Low confidence prediction. Capture a clearer leaf image.")

    print("\nPrevention:")
    print(prevention)

    print("\nTreatment:")
    print(treatment)

    print("\n==============================================")



    # ---------------------------------
    # DISPLAY WINDOWS
    # ---------------------------------

    cv2.imshow("Detected Leaf", leaf)

    cv2.imshow(
        "GradCAM Disease Region",
        cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()



# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python predict_tomato_final_pipeline.py image.jpg")

        exit()

    run_pipeline(sys.argv[1])