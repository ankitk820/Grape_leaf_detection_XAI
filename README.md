# 🍇 Grape Leaf Disease Classification using YOLOv11 + Grad-CAM

This app detects grape leaf diseases using YOLOv11 classification and visualizes decisions using Grad-CAM.

## 🔍 Classes
- Healthy
- Leaf Blight
- Black Measles
- Esca

## 🛠️ Model
Trained on custom dataset using YOLOv11 with 256x256 resolution.

## 🚀 Usage
1. Upload a grape leaf image.
2. See predicted disease.
3. View Grad-CAM heatmap.

## 👨‍💻 Run Locally
```bash
git clone https://github.com/yourusername/grape-leaf-disease-classification
cd grape-leaf-disease-classification
pip install -r requirements.txt
streamlit run app.py
