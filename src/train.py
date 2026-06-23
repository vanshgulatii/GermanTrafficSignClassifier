import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model import TrafficSignCNN

train_path = "../data/GTSRB/Final_Training/Images"

images = []
labels = []

classes = os.listdir(train_path)

print("Loading dataset...")

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

train_dataset = TensorDataset(
X_train,
y_train
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

model = TrafficSignCNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
model.parameters(),
lr=0.001
)

epochs = 30

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

print(
    f"Epoch {epoch+1}/{epochs} "
    f"Loss: "
    f"{running_loss / len(train_loader):.4f}"
)


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
100 * correct / total
)

print(
f"\nValidation Accuracy: "
f"{accuracy:.2f}%"
)

torch.save(
model.state_dict(),
"../models/traffic_sign_model.pth"
)

print(
"\nModel saved successfully."
)
