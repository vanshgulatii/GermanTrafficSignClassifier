import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Dataset
)

from torchvision import transforms

from model import TrafficSignCNN


# ======================
# Load Dataset
# ======================

train_path = "../data/GTSRB/Final_Training/Images"

images = []
labels = []

classes = sorted(
    os.listdir(train_path)
)

print("Loading dataset...")

train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.GaussianBlur(
        kernel_size=3
    ),

    transforms.ToTensor()
])

for class_id in classes:

    folder = os.path.join(
        train_path,
        class_id
    )

    for image_name in os.listdir(folder):

        if not image_name.endswith(".ppm"):
            continue

        image_path = os.path.join(
            folder,
            image_name
        )

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(
            img,
            (32, 32)
        )

        images.append(img)
        labels.append(
            int(class_id)
        )

X = np.array(images)
y = np.array(labels)

print("Images shape:", X.shape)
print("Labels shape:", y.shape)

# ======================
# Preprocessing
# ======================

X = X.astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

X_train = X_train.permute(
    0,
    3,
    1,
    2
)

X_val = X_val.permute(
    0,
    3,
    1,
    2
)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ======================
# Data Loaders
# ======================

class TrafficDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        transform=None
    ):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        image = image.permute(
            1,
            2,
            0
        ).numpy()

        if self.transform:
            image = self.transform(image)

        return image, label
    
train_dataset = TrafficDataset(
    X_train,
    y_train,
    transform=train_transform
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# ======================
# Model
# ======================

model = TrafficSignCNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

epochs = 40
best_acc = 0

train_losses = []
val_accuracies = []

# ======================
# Training Loop
# ======================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    train_losses.append(
        epoch_loss
    )

    # ======================
    # Validation
    # ======================

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100 *
        correct /
        total
    )

    val_accuracies.append(
        accuracy
    )

    # ======================
    # Save Best Model
    # ======================

    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            "../models/traffic_sign_model.pth"
        )

        print(
            "✅ Best model saved."
        )

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Accuracy: {accuracy:.2f}%"
    )

    scheduler.step()

# ======================
# Final Results
# ======================

print(
    f"\nBest Validation Accuracy: "
    f"{best_acc:.2f}%"
)

# ======================
# Save Training Loss Plot
# ======================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig(
    "../results/loss_curve.png"
)

# ======================
# Save Validation Accuracy Plot
# ======================

plt.figure(figsize=(8, 5))

plt.plot(
    val_accuracies
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")

plt.savefig(
    "../results/accuracy_curve.png"
)

print(
    "📈 Training curves saved in ../results/"
)