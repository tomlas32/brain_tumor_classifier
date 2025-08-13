import io
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# ---------------------------
# UI SETUP
# ---------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("🧠 Brain Tumor MRI Classifier — Streamlit App")
st.caption("Upload MRI scans and the corresponding class index mapping JSON to get model predictions.")

# ---------------------------
# HELPERS
# ---------------------------
def resize_and_pad_pil(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = Image.LANCZOS if scale < 1 else Image.BICUBIC
    img = img.resize((new_w, new_h), interp)
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    new_img.paste(img, (left, top))
    return new_img

def get_default_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(lambda im: resize_and_pad_pil(im.convert("RGB"), img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@st.cache_resource(show_spinner=False)
def load_model(num_classes: int, model_weights: bytes | None = None, architecture: str = "resnet18") -> torch.nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported architecture.")

    if model_weights is not None:
        buffer = io.BytesIO(model_weights)
        state = torch.load(buffer, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            st.error(f"Model weight mismatch. Missing: {missing}, Unexpected: {unexpected}")
            st.stop()

    model.eval()
    return model

def to_device(model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

def softmax_numpy(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits, dim=1).detach().cpu().numpy()

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Configuration")
arch = st.sidebar.selectbox("Architecture", ["resnet18", "resnet34"], index=0)
st.sidebar.caption("Image size is fixed to match training settings to avoid mismatches.")
img_size = 224  # Fixed to match training preprocessing

class_file = st.sidebar.file_uploader("Class index mapping (index_remap.json)", type=["json"], help="JSON mapping of class_name:index.")
model_file = st.sidebar.file_uploader("Model weights (.pth/.pt)", type=["pth", "pt"])

if not class_file:
    st.warning("Please upload index_remap.json to proceed.")
    st.stop()

# Load classes from JSON mapping
class_mapping = json.load(class_file)
class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

# ---------------------------
# LOAD MODEL
# ---------------------------
with st.spinner("Loading model…"):
    model = load_model(num_classes=len(class_names), model_weights=model_file.read() if model_file else None, architecture=arch)
    model, device = to_device(model)

# ---------------------------
# IMAGE UPLOAD
# ---------------------------
st.subheader("1) Upload MRI scans")
files = st.file_uploader("Drop multiple images here (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not files:
    st.info("Upload one or more images to get predictions.")
    st.stop()

# ---------------------------
# PREDICTION
# ---------------------------
st.subheader("2) Predictions")
transform = get_default_transform(img_size)

pred_rows = []
for f in files:
    pil = Image.open(f)
    proc = transform(pil)
    xb = proc.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xb)
    probs = softmax_numpy(logits)
    top_idx = int(np.argmax(probs[0]))
    top_prob = float(probs[0, top_idx])

    row = {"filename": f.name, "pred_class": class_names[top_idx], "confidence": round(top_prob, 4)}
    for i, c in enumerate(class_names):
        row[f"p({c})"] = round(float(probs[0, i]), 4)
    pred_rows.append(row)

import pandas as pd
st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Classes loaded from index_remap.json. Ensure it matches the training order.")
