# 🚦 German Traffic Sign Classifier

A deep learning-based traffic sign recognition system built using **PyTorch** and the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The project includes model training, prediction scripts, and a Streamlit web application for real-time traffic sign classification.

---

## 📌 Features

- Train a Convolutional Neural Network (CNN) on the GTSRB dataset
- Predict traffic signs from uploaded images
- Interactive web application using Streamlit
- Model saving and loading with PyTorch
- Real-world image testing
- Modular project structure suitable for further development

---

## 🖼️ Demo

### Streamlit Web App
Upload a traffic sign image and the model predicts:

- Traffic sign class
- Confidence score

Example:

| Image | Prediction |
|-------|-------------|
| 30 km/h speed sign | Speed limit (30km/h) |
| Confidence | 82.27% |

---

## 📂 Project Structure

```text
GermanTrafficSignClassifier/
│
├── app/
│   └── app.py
│
├── data/
│   └── GTSRB/
│
├── models/
│   └── traffic_sign_model.pth
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── dataset.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Model Architecture

```text
Input (3 × 32 × 32)
        ↓
Conv2D (32 filters)
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D (64 filters)
        ↓
ReLU
        ↓
MaxPool
        ↓
Fully Connected (128)
        ↓
Dropout (0.5)
        ↓
Output Layer (43 classes)
```

---

## 📊 Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

- Number of classes: 43
- Total images: 39,252
- Image size: 32 × 32 pixels

Dataset:

https://benchmark.ini.rub.de/gtsrb_news.html

---

## 📈 Results

| Metric | Value |
|---------|--------|
| Training Images | 39,252 |
| Number of Classes | 43 |
| Validation Accuracy | 87.99% |
| Framework | PyTorch |
| Deployment | Streamlit |

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/vanshgulatii/GermanTrafficSignClassifier.git
cd GermanTrafficSignClassifier
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
cd src
python train.py
```

The trained model will be saved in:

```text
models/traffic_sign_model.pth
```

---

## 🔍 Predict an Image

```bash
cd src

python predict.py \
../data/GTSRB/Final_Training/Images/00001/00001_00000.ppm
```

Example output:

```text
🚦 Traffic Sign Prediction
-----------------------------------
Sign       : Speed limit (30km/h)
Class ID   : 1
Confidence : 82.27%
```

---

## 🌐 Run the Web App

```bash
cd app
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🛠 Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-Learn
- Streamlit
- Git & GitHub

---

## 🔮 Future Improvements

- Data augmentation
- Top-3 predictions
- Confusion matrix visualization
- Transfer learning using ResNet18
- Model deployment to Streamlit Community Cloud
- Mobile-friendly interface

---

## 📸 Screenshots

Add screenshots here:

```text
images/app_homepage.png
images/prediction_example.png
```

---

## 👨‍💻 Author

**Vansh Gulati**

M.Sc. Digital Engineering  
Bauhaus-Universität Weimar  
Interested in Artificial Intelligence, Machine Learning and Computer Vision.

GitHub:
https://github.com/vanshgulatii

LinkedIn:
https://www.linkedin.com/in/vansh-g-31a607197/

---

## ⭐ If you found this project useful, consider giving it a star!