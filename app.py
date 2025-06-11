import streamlit as st
from PIL import Image
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load trained YOLOv11 model
@st.cache(allow_output_mutation=True)
def load_model():
    return YOLO("best.pt")

model = load_model()

# Class names (should match the ones in your dataset)
class_names = list(model.names.values())

st.title("🍇 Grape Leaf Disease Detection using YOLOv11")
st.markdown("""
Welcome to the **Grape Leaf Disease Detector**!  
This app uses a YOLOv11-based deep learning model to **detect diseases in grape leaves** from an uploaded image.

**Supported Classes:**  
- Black measles  
- Black rot  
- Leaf blight  
- Healthy  
""")
st.write("Upload an image of a grape leaf to detect the disease.")


st.title("🧠 Explainable AI: Grad-CAM on Grape Leaf Classifier")

uploaded_file = st.file_uploader("Upload a grape leaf image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    rgb_img = np.array(image) / 255.0
    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor.requires_grad_(True)

    result = model(image)[0]
    pred_class_id = int(result.probs.top1)

    cam = GradCAM(model=classifier_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class_id)])[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.image(visualization, caption=f"Grad-CAM Heatmap (Class: {pred_class_id})")
