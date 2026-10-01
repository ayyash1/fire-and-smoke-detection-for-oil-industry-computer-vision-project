# 🔥 Fire & Smoke Detection System for Oil & Gas Industry
### YOLOv8 • Computer Vision • Streamlit Dashboard

[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
This project presents an **AI-based Fire and Smoke Detection System** specifically designed for **oil & gas and industrial safety environments**. Utilizing a **custom-trained YOLOv8 object detection model**, the system identifies hazards in images, videos, and live camera feeds. Results are visualized through a professional-grade **Streamlit web interface** for real-time monitoring.

---

## 🎯 Objectives
* **Early Detection:** Identify fire and smoke hazards before they escalate.
* **Risk Mitigation:** Reduce the likelihood of industrial accidents and equipment loss.
* **Real-time Monitoring:** Provide safety officers with a live visual feed and automated alerts.
* **Practical AI:** Demonstrate the application of **Digital Image Processing** in hazardous industrial zones.

---

## 🏭 Application Areas
* **Oil & Gas Refineries:** Monitoring pipeline junctions and storage tanks.
* **Robotic Welding Cells:** Safety monitoring for automated welding sparks and flare-ups.
* **Chemical Plants:** Detecting early-stage chemical fires.
* **Power Generation Units:** Surveillance of high-voltage transformers and turbines.
* **Smart Surveillance:** Enhancing traditional CCTV with AI intelligence.

---

## 🧠 Model & Dataset
* **Architecture:** YOLOv8 (Ultralytics)
* **Dataset:** fire and smoke Computer Vision Dataset - Roboflow
* **Classes:** `Fire`, `Smoke`
* **Training Environment:** Google Colab (Tesla T4 GPU)
* **Model Weight File:** `best.pt`

---

## 📂 Project Structure
```text
fire-smoke-detection/
│
├── app.py              # Main Streamlit application
├── best.pt             # Trained YOLOv8 model weights
├── train.ipynb         # Google Colab training notebook
├── requirements.txt    # Python dependencies
├── samples/            # Test images and videos
└── README.md           # Project documentation
```
## ⚙️ Installation & Setup
1️⃣ Clone the Repository

```
git clone [https://github.com/your-username/fire-smoke-detection.git](https://github.com/your-username/fire-smoke-detection.git)
cd fire-smoke-detection
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
Note: Ensure you have Python 3.8+ installed.

## 🚀 Running the Application
▶️ Local Deployment

Launch the dashboard on your local machine:

```
streamlit run app.py
```
▶️ Google Colab Deployment

If running on Colab, use pyngrok to expose the local port:
Python
```
!pip install streamlit pyngrok
!streamlit run app.py & npx localtunnel --port 8501
```
## 🖥️ Dashboard Features

    Industrial UI: Professional dark-themed interface for low-light control rooms.

    Multi-Source Input: Seamlessly switch between Image upload, Video upload, or Live Webcam.

    Sensitivity Control: Adjustable confidence threshold (0.0 - 1.0) to reduce false positives.

    Live Inference: Real-time bounding box rendering and class labeling.

## 📊 Training Performance

The model was evaluated based on the following metrics:

    Precision & Recall: High detection accuracy with minimal false alarms.

    mAP@50: Mean Average Precision at 0.5 IoU threshold.

    Loss Curves: Box Loss, Class Loss, and DFL Loss monitored during training.

## 🔐 Safety Disclaimer

This project is intended for academic and research purposes only. While YOLOv8 is highly capable, industrial-grade deployment requires integration with certified fire suppression systems, thermal imaging hardware, and redundant safety protocols.

## 👨‍💻 Author

Ahamed Ayyash Computer Engineering Student Specialization: AI • Machine Learning • Computer Vision

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⭐ Acknowledgements

    Ultralytics for the YOLOv8 framework.

    Roboflow for dataset management.

    Streamlit for the amazing web interface library.
