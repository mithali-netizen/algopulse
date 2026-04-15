# 💙 AlgoPulse AI: Breast Cancer Detection System

> Making AI-powered diagnosis **understandable, accessible, and assistive** for both doctors and patients.

---

## 🌟 Overview

ALGOPULSE AI is a web-based application that uses **deep learning and explainable AI** to assist in breast cancer detection using ultrasound images.

Unlike traditional AI systems that act as black boxes, ALGOPULSE focuses on:

* 🧠 Accurate predictions
* 🔍 Transparent explanations
* 👩‍⚕️ Clinical support for doctors
* 🗣️ Easy understanding for patients

---

## 🚀 Key Highlights

✨ **AI Diagnosis using EfficientNet-B0**
✨ **Visual Explainability with Grad-CAM**
✨ **Doctor Dashboard with Similar Case Retrieval (Qdrant)**
✨ **Patient Dashboard with Voice Assistant (Vapi)**
✨ **Fast and Interactive Web Application**

---

## 🧠 How It Works

1. User uploads an ultrasound image
2. AI model predicts:

   * Benign / Malignant / Normal
3. Grad-CAM highlights important regions
4. System provides:

   * 📊 Confidence score
   * 🔍 Visual explanation
   * 📁 Similar past cases (Doctor view)
   * 🎤 Voice explanation (Patient view)

---

## 🧑‍⚕️ Dual Dashboard System

### 🩺 Doctor Dashboard

* Detailed prediction results
* Probability distribution
* Grad-CAM visualization
* 🔎 Similar case retrieval using Qdrant

### 👩 Patient Dashboard

* Simplified results
* Easy-to-understand explanations
* 🎙️ Voice assistant powered by Vapi

---

## 🏗️ Architecture

```
Image Input
    ↓
EfficientNet Model
    ↓
Prediction + Embedding
   ↓             ↓
Grad-CAM      Qdrant (Similarity Search)
                    ↓
              Similar Cases
                    ↓
               Frontend UI
        ↓                     ↓
 Doctor Dashboard      Patient Dashboard

```

---

## 🛠️ Tech Stack

### 🧠 AI / ML

* EfficientNet-B0 (PyTorch)
* Grad-CAM

### ⚙️ Backend

* Flask (Python)

### 💻 Frontend

* React + JavaScript

### 🗄️ Database

* Qdrant (Vector Database)

### 🎙️ Voice AI

* Vapi

### 🧰 Tools

* VS Code
* GitHub
* Postman

---

## 📁 Project Structure

```
ETHiCARE/
├── backend/          # Flask API + ML model
├── algopulse-ui/     # Main React frontend
├── frontend/         # Alternative UI
├── dataset/          # Training data
```

---

## 🚀 Getting Started

### 1️⃣ Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 2️⃣ Frontend

```bash
cd algopulse-ui
npm install
npm start
```

---

## 📊 Model Performance

* Accuracy: **94.2%**
* Precision: **93.8%**
* Recall: **94.5%**

---

## 🔮 Future Improvements

* Use full dataset with scalable Qdrant deployment
* Add patient history tracking
* Improve model with larger datasets
* Deploy on cloud for real-world use

---

## ⚠️ Disclaimer

This project is for **educational and assistive purposes only**.
It should not replace professional medical diagnosis.

---
