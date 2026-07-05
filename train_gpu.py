import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from PIL import ImageFile

# Fix corrupted image crash
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================
# CONFIG
# =============================
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0003
MODEL_PATH = "tomato_disease_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =============================
# DATA AUGMENTATION
# =============================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),

    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),

    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =============================
# DATASETS
# =============================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR,"train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR,"valid"),
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR,"test"),
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

num_classes = len(train_dataset.classes)

print("Number of classes:", num_classes)

# Save class names
torch.save(train_dataset.classes,"classes.pth")

print("Saved class names")

# =============================
# MODEL
# =============================

model = models.efficientnet_b0(weights="IMAGENET1K_V1")

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# =============================
# LOSS + OPTIMIZER
# =============================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

scaler = torch.amp.GradScaler("cuda")

train_acc_history = []
val_acc_history = []

best_val_acc = 0

# =============================
# TRAINING LOOP
# =============================

for epoch in range(EPOCHS):

    model.train()

    correct = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        _, preds = torch.max(outputs,1)

        correct += torch.sum(preds == labels)

    train_acc = correct.double()/len(train_dataset)

    train_acc_history.append(train_acc.cpu().numpy())

    # =============================
    # VALIDATION
    # =============================

    model.eval()

    correct = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):

                outputs = model(images)

            _, preds = torch.max(outputs,1)

            correct += torch.sum(preds == labels)

    val_acc = correct.double()/len(val_dataset)

    val_acc_history.append(val_acc.cpu().numpy())

    scheduler.step(val_acc)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    print(f"Train Accuracy: {train_acc:.4f}")

    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(), MODEL_PATH)

        print("Best model saved")

# =============================
# TEST EVALUATION
# =============================

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

all_preds=[]
all_labels=[]

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)

        with torch.amp.autocast("cuda"):

            outputs = model(images)

        _,preds = torch.max(outputs,1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.numpy())

print("\nClassification Report\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=train_dataset.classes
))

# =============================
# CONFUSION MATRIX
# =============================

cm = confusion_matrix(all_labels,all_preds)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=False,
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()

# =============================
# ACCURACY GRAPH
# =============================

plt.plot(train_acc_history,label="Train Accuracy")

plt.plot(val_acc_history,label="Validation Accuracy")

plt.legend()

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()