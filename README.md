# Real-Time & Image Face Emotion Recognition 🎭

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-AI-orange.svg)](https://github.com/serengil/deepface)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight and powerful computer vision application built with **OpenCV** and **DeepFace** for detecting faces and classifying facial emotions in real-time video streams (webcam) or static images.

---

## 🌟 Key Features

- **Real-Time Webcam Analysis**: Live facial emotion detection overlay on webcam feeds with high frame-rate processing (`app.py`).
- **Static Image Analysis**: CLI tool to analyze static images, compute confidence percentages across emotion categories, and export annotated images (`image_demo.py`).
- **Multi-Emotion Classification**: Detects 7 core facial emotions: *Angry*, *Disgust*, *Fear*, *Happy*, *Sad*, *Surprise*, and *Neutral*.
- **Interactive Jupyter Notebook**: Step-by-step notebook demonstration (`emotion_recognition_demo.ipynb`).
- **Flexible Backends**: Supports multiple face detector backends (`opencv`, `retinaface`, `mtcnn`, `ssd`, `yolov8`).

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Computer Vision**: OpenCV (`opencv-python`)
- **Deep Learning / Facial Analysis**: DeepFace (`deepface`), TensorFlow / Keras (`tf-keras`)
- **Data Visualization**: Matplotlib

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Face_Emotion_Recognisation.git
cd Face_Emotion_Recognisation
```

### 2. Create a Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: On initial run, DeepFace will automatically download pre-trained lightweight emotion classification model weights into your home directory (`~/.deepface/weights/`).

---

## 💻 Usage Instructions

### 🎥 1. Real-Time Webcam Stream (`app.py`)
Launch live webcam emotion recognition:
```bash
python app.py
```

**CLI Command Options**:
```bash
# Specify camera index (default: 0)
python app.py --device 1

# Change face detector backend (e.g., retinaface, mtcnn, opencv)
python app.py --detector retinaface
```
> Press **`q`** in the video window at any time to exit.

---

### 🖼️ 2. Static Image Analysis (`image_demo.py`)
Run emotion recognition on a single image file:
```bash
# Analyze a sample image and view annotated output
python image_demo.py --image samples/happy_boy.jpg

# Save the annotated output image to disk
python image_demo.py --image samples/sad_woman.jpg --output output_sad.jpg
```

---

### 📓 3. Interactive Jupyter Notebook (`emotion_recognition_demo.ipynb`)
Open and run the step-by-step Jupyter Notebook:
```bash
jupyter notebook emotion_recognition_demo.ipynb
```

---

## 📁 Project Structure

```text
Face_Emotion_Recognisation/
├── app.py                      # Real-time webcam emotion recognition script
├── image_demo.py                # Command-line image analysis script
├── emotion_recognition_demo.ipynb # Jupyter Notebook walkthrough
├── samples/                    # Sample test images
│   ├── happy_boy.jpg
│   ├── sad_woman.jpg
│   ├── sample_person1.jpeg
│   └── sample_person2.jpeg
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclude patterns
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

1. **Face Detection**: The frame or image is processed through OpenCV or a specified detector backend (`retinaface`/`mtcnn`) to locate facial bounding boxes.
2. **Preprocessing**: The cropped face is normalized and converted to the standard target size expected by the emotion neural network.
3. **Emotion Prediction**: DeepFace passes the face image through a pre-trained Convolutional Neural Network (CNN) model to output softmax confidence scores across 7 emotion categories.
4. **Overlay & Rendering**: Bounding box coordinates and the dominant emotion with confidence percentage are drawn onto the frame using OpenCV.

---

## 🔧 Troubleshooting

- **macOS Camera Permission Error**: Ensure Terminal or your IDE (VS Code, PyCharm, iTerm) has permission to access the Camera under `System Settings` &rarr; `Privacy & Security` &rarr; `Camera`.
- **First-run delay**: DeepFace downloads pre-trained model weights on the first run. Ensure you have an active internet connection on the initial launch.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
