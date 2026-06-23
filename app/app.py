import streamlit as st
import torch
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

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
    43: "Unknown traffic sign"
}

@st.cache_resource
def load_model():
    model = TrafficSignCNN()

    MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "traffic_sign_model.pth"
    )
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
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

        top_probs, top_classes = torch.topk(
        probabilities,
        3
        )


    top_confidence = top_probs[0][0].item()

    return (
        top_classes[0].tolist(),
        top_probs[0].tolist(),
        top_confidence
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

st.caption(
    "Model trained on the German Traffic Sign Recognition Benchmark (GTSRB). Predictions on unseen traffic signs may be unreliable."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "ppm"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    predictions, confidences, top_confidence = predict_image(
    image
    )

    margin = confidences[0] - confidences[1]

    if top_confidence < 0.80 or margin < 0.10:
        st.warning(
            "⚠️ The model is uncertain. This may be an unseen traffic sign."
        )

    if top_confidence >= 0.80 and margin >= 0.10:
        st.success(
            "✅ High confidence prediction"
        )

    st.subheader("Top Predictions")

    for i in range(3):

        class_id = predictions[i]
        confidence = confidences[i]

        st.write(
            f"{i+1}. "
            f"{class_names[class_id]}"
            f" — "
            f"{confidence*100:.2f}%"
        )