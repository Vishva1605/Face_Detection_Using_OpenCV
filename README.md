# Face Detection using OpenCV
<div align="center">

# 👤 Face Detection & Recognition using OpenCV  
### Python • Haar Cascade • LBPH Face Recognizer

<img src="https://img.shields.io/badge/Python-3.8%20--%203.11-blue" />
<img src="https://img.shields.io/badge/OpenCV-Contrib-success" />
<img src="https://img.shields.io/badge/Face%20Detection-Haar%20Cascade-orange" />
<img src="https://img.shields.io/badge/Face%20Recognition-LBPH-purple" />

</div>

---

## 📌 Project Overview

This project implements a **Face Detection and Face Recognition system** using **OpenCV in Python**.

- **Face Detection** is performed using **Haar Cascade Classifier**
- **Face Recognition** is implemented using **LBPH (Local Binary Patterns Histogram)**

The system can:
- Train faces from a dataset
- Save the trained face recognition model
- Detect and recognize faces in **real-time using webcam**

---

## 📂 Project Structure

FACEDetection/
│
├── dataset/
│
├── benchmark.py
├── detect_face_video.py
├── haarcascade_frontalface_alt2.xml
├── labels.txt
├── README.md
├── requirements.txt
├── test.jpg
├── train_faces.py
└── trained_model.yml

---

## 🧠 How It Works

### 1️⃣ Dataset
- Each folder inside `dataset/` represents **one person**
- Folder name = **person name**
- Images inside are used for training

### 2️⃣ Training
- `train_faces.py`:
  - Reads images from dataset
  - Detects faces using Haar Cascade
  - Trains LBPH face recognizer
  - Saves:
    - `trained_model.yml`
    - `labels.txt`

### 3️⃣ Recognition
- `detect_face_video.py`:
  - Opens webcam
  - Detects faces in real time
  - Recognizes trained faces using saved model

---

## ⚙️ Requirements

- Python **3.8 – 3.11** (Recommended)
- OpenCV Contrib
- NumPy
- Pillow

> ⚠️ Python 3.13 is **not recommended** for OpenCV face recognition.

---
## 📸 Output Screenshot

<p align="center">
  <img src="output/result.png" alt="Face Detection Output" width="700">
</p>


## 📦 Installation

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv

## Install Dependencies

pip install -r requirements.txt
pip install opencv-contrib-python

## 🚀 Usage
🔹 Train the Face Recognition Model
python train_faces.py


This will:

Train faces from dataset/

Generate:

trained_model.yml

labels.txt

🔹 Run Real-Time Face Detection & Recognition

python detect_face_video.py

Opens webcam
Detects and recognizes faces in real time
Press q to exit

📸 Dataset Guidelines

Each person must have a separate folder
Use multiple images per person
Images should be:
Clear face images
Different angles
Proper lighting

Example:

dataset/
└── person_name/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg


📜 Disclaimer

This project is developed for educational and learning purposes only.
</div> 

## 📜 License
This project is licensed under the MIT License.
