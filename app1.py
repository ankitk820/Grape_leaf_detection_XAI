import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load YOLOv11 model
model = YOLO("best.pt")
classifier_model = model.model.eval()
target_layers = [classifier_model.model[-2]]
class_names = list(model.names.values())

# App UI
st.title("🍇 Grape Leaf Disease Detector with Grad-CAM")
st.markdown("Upload an image of a **grape leaf** to identify its disease class.")

uploaded_file = st.file_uploader("📤 Upload a grape leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    rgb_img = np.array(image) / 255.0
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to tensor
    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor.requires_grad = True

    # Run inference
    result = model(image)[0]
    pred_class_id = int(result.probs.top1)
    pred_class_name = class_names[pred_class_id]
    confidence = result.probs.data[pred_class_id].item()

    # Check confidence threshold
    CONF_THRESHOLD = 0.5  # Adjust as needed
    if confidence < CONF_THRESHOLD:
        st.error("⚠️ This image does not appear to be a grape leaf or is unclear. Please upload a valid grape leaf image.")
    else:
        # Show prediction
        st.success(f"✅ Prediction: **{pred_class_name}** (Confidence: {confidence:.2f})")

        # Grad-CAM visualization
        cam = GradCAM(model=classifier_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class_id)])[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
