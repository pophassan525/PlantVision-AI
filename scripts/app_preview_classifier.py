"""
PlantVision AI — Plant Classifier Preview App (TEMPORARY)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))
from src.classification.plant_classifier import build_model

CHECKPOINT_PATH = Path("models/plant_classifier/best_model.pt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="PlantVision AI — Classifier Preview", page_icon="🌿")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found at {CHECKPOINT_PATH}. "
                 f"Run training/train_plant_classifier.py first.")
        st.stop()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, idx_to_class, device, checkpoint


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def format_class_name(raw: str) -> str:
    parts = raw.split("___")
    if len(parts) == 2:
        plant, condition = parts
        return f"{plant.replace('_', ' ')} — {condition.replace('_', ' ')}"
    return raw.replace("_", " ")


st.title("🌿 PlantVision AI")
st.caption(
    "Preview build — plant identification / disease classifier only."
)

model, idx_to_class, device, checkpoint = load_model()
st.sidebar.markdown("### Model info")
st.sidebar.write(f"Device: `{device}`")
st.sidebar.write(f"Checkpoint epoch: {checkpoint['epoch']}")
st.sidebar.write(f"Training-time val_acc: {checkpoint['val_acc']:.4f}")
st.sidebar.write(f"Classes: {len(idx_to_class)}")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()

    top5_probs, top5_idx = torch.topk(probs, k=min(5, len(idx_to_class)))

    with col2:
        st.subheader("Prediction")
        top_class = format_class_name(idx_to_class[top5_idx[0].item()])
        top_conf = top5_probs[0].item()
        st.markdown(f"**{top_class}**")
        st.markdown(f"Confidence: **{top_conf:.1%}**")

        st.subheader("Top 5 candidates")
        for prob, idx in zip(top5_probs, top5_idx):
            label = format_class_name(idx_to_class[idx.item()])
            st.write(label)
            st.progress(prob.item())

    st.info("This is a raw classifier output for one image — not a full plant health report.")
else:
    st.write("Upload an image above to try the trained classifier.")