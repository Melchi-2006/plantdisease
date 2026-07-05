import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==============================
# Load Model
# ==============================

MODEL_PATH = "tomato_disease_model.pth"
CLASSES_PATH = "classes.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = torch.load(CLASSES_PATH)

num_classes = len(classes)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

# ==============================
# Image Preprocessing
# ==============================

image_path = "test_leaf.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rgb_img = np.float32(image) / 255

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

input_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

# ==============================
# Prediction
# ==============================

outputs = model(input_tensor)
_, pred = torch.max(outputs,1)

pred_class = classes[pred.item()]

print("Predicted Disease:", pred_class)

# ==============================
# GradCAM Setup
# ==============================

target_layers = [model.features[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [ClassifierOutputTarget(pred.item())]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

# ==============================
# Show Result
# ==============================

cv2.imshow("GradCAM Heatmap", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()