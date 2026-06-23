import streamlit as st
import torch
import sys
sys.path.append("../src")

from model import TrafficSignCNN
import cv2
import numpy as np
from PIL import Image


class_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice or snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

@st.cache_resource
def load_model():
    model = TrafficSignCNN()

    model.load_state_dict(
        torch.load(
            "models/traffic_sign_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


def predict_image(image):

    img = np.array(image)

    # Handle PNGs with transparency
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (32, 32))

    img = img.astype(np.float32) / 255.0

    img = torch.tensor(
        img,
        dtype=torch.float32
    )

    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        prediction.item(),
        confidence.item()
    )


st.set_page_config(
    page_title="German Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 German Traffic Sign Classifier")

st.write(
    "Upload a traffic sign image and let the AI identify it."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "ppm"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    prediction, confidence = predict_image(
        image
    )

    st.success(
        f"Prediction: {class_names[prediction]}"
    )

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )