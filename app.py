import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load model
model = YOLO("best.pt")
classifier_model = model.model.eval()
target_layers = [classifier_model.model[-2]]
class_names = list(model.names.values())

st.title("🍇 Grape Leaf Disease Detector with Grad-CAM")

uploaded_file = st.file_uploader("Upload a grape leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    rgb_img = np.array(image) / 255.0
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor.requires_grad = True

    result = model(image)[0]
    pred_class_id = int(result.probs.top1)
    pred_class_name = class_names[pred_class_id]
    st.success(f"Prediction: **{pred_class_name}**")

    # Grad-CAM
    cam = GradCAM(model=classifier_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class_id)])[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
