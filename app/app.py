"""
Brain Tumor MRI — Inference App (Streamlit)

Features
--------
- Loads trained ResNet (18/34/50) weights from your pipeline.
- Robust class mapping loader for index_remap.json (supports {idx: name} or {name: idx}).
- ImageNet preprocessing at 224px (to match training).
- Device auto-select (GPU if available).
- Displays per-class probabilities and top-1 prediction.

Author: Tomasz Lasota
Date: 2025-08-16
Version: 1.0
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms as T


# --------------------------- Logging -----------------------------------------

log = logging.getLogger("app.inference")
if not log.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    log.addHandler(handler)
log.setLevel(logging.INFO)


# --------------------------- Utils -------------------------------------------

def normalize_class_names(mapping: Dict) -> List[str]:
    """
    Normalize class names from a JSON mapping.

    Accepts either:
    - idx -> name    e.g., {"0": "Glioma", "1": "Meningioma", ...}
    - name -> idx    e.g., {"Glioma": 0, "Meningioma": 1, ...}

    Returns
    -------
    list[str]: Ordered class names by index.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("index_remap.json must be a non-empty object.")

    keys = list(mapping.keys())
    vals = list(mapping.values())

    # idx -> name (preferred by pipeline)
    if all(str(k).isdigit() for k in keys):
        return [name for _, name in sorted(mapping.items(), key=lambda kv: int(kv[0]))]

    # name -> idx (legacy)
    if all(isinstance(v, int) or str(v).isdigit() for v in vals):
        return [name for name, idx in sorted(mapping.items(), key=lambda kv: int(kv[1]))]

    raise ValueError("Unrecognized index_remap format. Expected {idx: name} or {name: idx}.")


def build_preprocess(image_size: int = 224) -> T.Compose:
    """
    Preprocessing to mirror training/evaluation:
    - Resize shorter side to `image_size` and center-crop (or direct resize to square).
    - Convert to tensor.
    - Normalize with ImageNet stats.
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def build_model(architecture: str, num_classes: int) -> nn.Module:
    """
    Build a torchvision ResNet and replace the classifier head for `num_classes`.
    """
    architecture = architecture.lower().strip()
    if architecture == "resnet18":
        m = models.resnet18(weights=None)
    elif architecture == "resnet34":
        m = models.resnet34(weights=None)
    elif architecture == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def load_state_dict_flexible(model: nn.Module, weight_bytes: bytes, device: torch.device) -> None:
    """
    Load weights from uploaded bytes. Accepts:
    - raw state_dict
    - wrapper dict with 'state_dict' key
    - keys possibly prefixed with 'module.' (DataParallel)

    Uses strict=True where possible; falls back to strict=False with a warning.
    """
    buf = io.BytesIO(weight_bytes)
    sd = torch.load(buf, map_location=device)

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # Strip 'module.' if present
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[len("module."):]] = v
            else:
                new_sd[k] = v
        sd = new_sd

    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        log.warning("Strict load failed, retrying with strict=False. Error=%s", str(e))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            log.warning("Missing keys: %s | Unexpected keys: %s", missing, unexpected)


def predict_one(model: nn.Module, device: torch.device, img_pil: Image.Image, tf: T.Compose) -> Tuple[int, np.ndarray]:
    """
    Run a single-image prediction.

    Returns
    -------
    (pred_idx, probs_np)
    """
    model.eval()
    with torch.inference_mode():
        xb = tf(img_pil).unsqueeze(0).to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return pred_idx, probs


# --------------------------- Streamlit UI ------------------------------------

st.set_page_config(page_title="Brain Tumor MRI — Inference", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor MRI — Inference")
st.caption("Upload your trained weights and index_remap.json, then drop MRI images to classify.")

with st.sidebar:
    st.header("Configuration")
    arch = st.selectbox("Architecture", ["resnet18", "resnet34", "resnet50"], index=0)

    st.markdown("**Image size** is fixed to 224 to match training.")
    image_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Device: `{device.type}`")

    weights_file = st.file_uploader("Model weights (.pth/.pt)", type=["pth", "pt"])
    mapping_file = st.file_uploader("Class mapping (index_remap.json)", type=["json"])

st.divider()

# Inputs validation
if not weights_file:
    st.warning("Please upload **trained model weights** (.pth or .pt) in the sidebar to enable inference.")
    st.stop()

if not mapping_file:
    st.warning("Please upload **index_remap.json** (class mapping) in the sidebar.")
    st.stop()

# Load classes
try:
    class_mapping = json.load(mapping_file)
    class_names = normalize_class_names(class_mapping)
except Exception as e:
    st.error(f"Failed to parse class mapping: {e}")
    st.stop()

# Build model
num_classes = len(class_names)
try:
    model = build_model(arch, num_classes)
except Exception as e:
    st.error(f"Failed to build model: {e}")
    st.stop()

# Load weights
try:
    load_state_dict_flexible(model, weights_file.read(), device)
    model = model.to(device)
except Exception as e:
    st.error(f"Failed to load weights: {e}")
    st.stop()

# Preprocess
preprocess = build_preprocess(image_size=image_size)

# Sanity display
with st.expander("Class Index → Name mapping", expanded=False):
    st.write({i: name for i, name in enumerate(class_names)})

# Image uploader
st.subheader("Upload images")
uploads = st.file_uploader("Select one or more MRI images", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=True)

if not uploads:
    st.info("No images uploaded yet.")
    st.stop()

# Predict each image
for f in uploads:
    try:
        img = Image.open(f).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image `{f.name}`: {e}")
        continue

    pred_idx, probs = predict_one(model, device, img, preprocess)
    top1_name = class_names[pred_idx]
    top1_prob = float(probs[pred_idx])

    # Layout
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(img, caption=f.name, use_container_width=True)
    with col2:
        st.markdown(f"### Prediction: **{top1_name}**")
        st.write(f"Confidence: **{top1_prob:.3f}**")
        # Prob table
        st.markdown("**Per-class probabilities**")
        # Build a small table sorted by prob desc
        order = np.argsort(-probs)
        rows = [{"class": class_names[i], "probability": float(probs[i])} for i in order]
        st.dataframe(rows, use_container_width=True, hide_index=True)

st.success("Done.")
