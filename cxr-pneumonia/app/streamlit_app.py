from __future__ import annotations

import os
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image

import sys
from pathlib import Path

# Allow running directly: add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from models import create_model, load_checkpoint
from transforms import get_transforms


st.set_page_config(page_title="CXR Pneumonia Detection", layout="centered")
st.title("🩺 Chest X-ray Pneumonia Classifier")
st.caption("Upload an X-ray and get a pneumonia probability (for research/education only).")


@st.cache_resource
def load_model(weights_path: str, model_name: str, image_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(name=model_name, num_classes=1, pretrained=False).to(device)
    model = load_checkpoint(weights_path, model, device)
    model.eval()
    return model, device


def predict_image(model, device, img: Image.Image, image_size: int) -> float:
    _, val_tfms = get_transforms(image_size=image_size, aug="none")
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = val_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        prob = torch.sigmoid(logits).item()
    return float(prob)


with st.sidebar:
    st.header("Settings")
    weights_path = st.text_input("Weights path", value=str(ROOT / "experiments" / "best.ckpt"))
    model_name = st.selectbox("Backbone", ["resnet18", "resnet34", "efficientnet_b0"], index=0)
    image_size = st.slider("Image size", 128, 512, 224, step=16)
    ready = st.button("Load model")

uploaded = st.file_uploader("Upload CXR image (PNG/JPG)", type=["png", "jpg", "jpeg"])    

if ready:
    if not os.path.exists(weights_path):
        st.warning("Weights not found. Train a model first and provide the correct path.")
    else:
        model, device = load_model(weights_path, model_name, image_size)
        st.success("Model loaded")
        if uploaded is not None:
            img = Image.open(BytesIO(uploaded.read()))
            st.image(img, caption="Uploaded image", use_column_width=True)
            prob = predict_image(model, device, img, image_size)
            st.metric("Pneumonia probability", f"{prob:.3f}")
else:
    st.info("Set weights and click 'Load model' to begin.")
