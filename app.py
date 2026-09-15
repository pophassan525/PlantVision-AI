"""
PlantVision AI — Streamlit Application (Green Field Edition)

A single-file app for plant leaf analysis:
  - Real model metrics pulled from reports/plant_classifier_metrics.json
  - Auto-crop leaf detection (OpenCV GrabCut)
  - Plant + disease prediction with confidence gauge
  - Per-disease information and care advice
  - Grad-CAM attention map
  - Full model info tab with real evaluation artifacts

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
from src.classification.plant_classifier import build_model  # noqa: E402

# ============================================================================
# Paths & constants
# ============================================================================
ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "plant_classifier" / "best_model.pt"
METRICS_PATH = ROOT / "reports" / "plant_classifier_metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "reports" / "plant_classifier_confusion_matrix.png"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Page config + Green Field theme
# ============================================================================
st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ---- Global ---- */
    .stApp {
        background: linear-gradient(180deg, #0f1a14 0%, #14261b 100%);
    }

    /* ---- Header ---- */
    .pv-hero {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(74, 222, 128, 0.15);
        border: 1px solid rgba(74, 222, 128, 0.2);
    }
    .pv-hero h1 {
        color: #d1fae5;
        font-size: 2.8rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .pv-hero p {
        color: #a7f3d0;
        font-size: 1.15rem;
        margin-top: 0.75rem;
        margin-bottom: 0;
        opacity: 0.95;
    }
    .pv-badge {
        display: inline-block;
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(74, 222, 128, 0.3);
        margin-top: 1rem;
    }

    /* ---- Result cards ---- */
    .pv-result {
        border-radius: 20px;
        padding: 2rem;
        margin: 0;
        border: 2px solid;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    .pv-result-healthy {
        background: linear-gradient(135deg, rgba(26, 77, 46, 0.6) 0%, rgba(45, 106, 79, 0.4) 100%);
        border-color: #4ade80;
    }
    .pv-result-diseased {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(153, 27, 27, 0.3) 100%);
        border-color: #f87171;
    }
    .pv-result-icon {
        font-size: 3.5rem;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .pv-result-plant {
        font-size: 2rem;
        font-weight: 700;
        color: #f0fdf4;
        margin: 0.25rem 0;
    }
    .pv-result-condition {
        font-size: 1.15rem;
        color: #d1fae5;
        margin: 0;
        opacity: 0.9;
    }
    .pv-result-diseased .pv-result-condition {
        color: #fecaca;
    }

    /* ---- Metric cards ---- */
    .pv-metric-card {
        background: rgba(26, 77, 46, 0.35);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
    }
    .pv-metric-label {
        color: #86efac;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .pv-metric-value {
        color: #f0fdf4;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }

    /* ---- Top-5 ranking ---- */
    .pv-rank-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(26, 77, 46, 0.25);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4ade80;
    }
    .pv-rank-item.first {
        background: rgba(74, 222, 128, 0.18);
        border-left-color: #22c55e;
    }
    .pv-rank-name {
        color: #f0fdf4;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .pv-rank-prob {
        color: #4ade80;
        font-weight: 700;
        font-size: 1rem;
    }

    /* ---- Info boxes ---- */
    .pv-info-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #4ade80;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .pv-info-box h4 {
        color: #4ade80;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    .pv-info-box p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* ---- Footer ---- */
    .pv-footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(74, 222, 128, 0.15);
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
    }
    .pv-footer strong { color: #86efac; }

    /* ---- Animations ---- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---- Streamlit tweaks ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 77, 46, 0.2);
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a7f3d0;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(74, 222, 128, 0.2) !important;
        color: #4ade80 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f 0%, #1a4d2e 100%);
        color: #f0fdf4;
        border: 1px solid #4ade80;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        color: #0f1a14;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# Model + metrics loading (cached)
# ============================================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found at {CHECKPOINT_PATH}.")
        st.stop()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, idx_to_class, device, ckpt


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
@st.cache_data
def load_training_log():
    """Load the training log CSV produced by train_plant_classifier.py."""
    path = ROOT / "reports" / "plant_classifier_training_log.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def count_dataset_images():
    """Count images per class in train / validation / test splits."""
    counts = {}
    for split in ["train", "validation", "test"]:
        split_dir = ROOT / "data" / split
        if not split_dir.exists():
            continue
        n = 0
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                n += sum(1 for _ in cls_dir.glob("*") if _.is_file())
        counts[split] = n
    return counts


@st.cache_data
def count_per_class():
    """Count training images for each class."""
    train_dir = ROOT / "data" / "train"
    if not train_dir.exists():
        return None
    rows = []
    for cls_dir in train_dir.iterdir():
        if cls_dir.is_dir():
            n = sum(1 for _ in cls_dir.glob("*") if _.is_file())
            rows.append({"class": cls_dir.name, "count": n})
    return pd.DataFrame(rows).sort_values("count", ascending=False)

# ============================================================================
# Disease info dictionary — brief descriptions + care advice for all 38 classes
# ============================================================================
DISEASE_INFO: dict[str, dict[str, str]] = {
    # ---- Apple ----
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "description": "Fungal disease causing olive-green to black spots on leaves; fruit may develop rough, scabby lesions.",
        "advice": "Remove fallen leaves, apply fungicide in early spring, improve air circulation.",
    },
    "Apple___Black_rot": {
        "severity": "Severe",
        "description": "Fungal disease causing brown rot on fruit and purple-bordered leaf spots; can kill branches.",
        "advice": "Prune infected branches, remove mummified fruit, apply fungicide during growing season.",
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "description": "Fungal disease causing bright orange-yellow spots on leaves; needs cedar trees nearby to complete its cycle.",
        "advice": "Remove nearby cedar hosts if possible, apply fungicide in spring.",
    },
    "Apple___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; maintain good watering and pruning practices.",
    },
    # ---- Blueberry ----
    "Blueberry___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep soil acidic (pH 4.5-5.5), water regularly, monitor for pests.",
    },
    # ---- Cherry ----
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "Moderate",
        "description": "Fungal disease producing white powdery coating on leaves; can stunt growth.",
        "advice": "Improve air circulation, avoid overhead watering, apply sulfur or potassium bicarbonate.",
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; ensure good drainage and sunlight.",
    },
    # ---- Corn (maize) ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing rectangular gray-to-tan lesions running parallel to leaf veins.",
        "advice": "Rotate crops, use resistant hybrids, apply fungicide if severe.",
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "description": "Fungal disease producing reddish-brown pustules on both leaf surfaces.",
        "advice": "Plant resistant hybrids, apply fungicide early if detected.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "Severe",
        "description": "Fungal disease causing long, cigar-shaped gray-green lesions on leaves.",
        "advice": "Rotate crops, use resistant hybrids, apply fungicide at first sign.",
    },
    "Corn_(maize)___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; ensure adequate nitrogen and water.",
    },
    # ---- Grape ----
    "Grape___Black_rot": {
        "severity": "Severe",
        "description": "Fungal disease causing brown leaf spots and black, shriveled fruit.",
        "advice": "Remove mummified berries, apply fungicide from bud break onward.",
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "Severe",
        "description": "Complex fungal disease causing interveinal chlorosis and dark spotting; can kill vines.",
        "advice": "Prune affected wood, avoid vine stress, no full cure once established.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "description": "Fungal disease causing irregular dark brown spots on leaves, often with yellow halos.",
        "advice": "Improve air circulation, remove fallen leaves, apply fungicide if severe.",
    },
    "Grape___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; maintain proper pruning and canopy management.",
    },
    # ---- Orange ----
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "Severe",
        "description": "Bacterial disease spread by psyllids; causes blotchy mottled leaves and misshapen fruit. No cure.",
        "advice": "Control psyllid vectors, remove infected trees, use certified disease-free stock.",
    },
    # ---- Peach ----
    "Peach___Bacterial_spot": {
        "severity": "Moderate",
        "description": "Bacterial disease causing small dark water-soaked spots on leaves, fruit, and twigs.",
        "advice": "Apply copper-based sprays, avoid overhead watering, plant resistant varieties.",
    },
    "Peach___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; ensure good drainage and annual pruning.",
    },
    # ---- Pepper, bell ----
    "Pepper,_bell___Bacterial_spot": {
        "severity": "Moderate",
        "description": "Bacterial disease causing dark water-soaked spots with yellow halos on leaves.",
        "advice": "Use disease-free seed, rotate crops, apply copper sprays if detected.",
    },
    "Pepper,_bell___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; water consistently and mulch to prevent stress.",
    },
    # ---- Potato ----
    "Potato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark brown spots with concentric rings (target-like) on lower leaves.",
        "advice": "Rotate crops, remove affected leaves, apply fungicide regularly.",
    },
    "Potato___Late_blight": {
        "severity": "Severe",
        "description": "Devastating fungal-like disease causing large water-soaked lesions; caused the Irish potato famine.",
        "advice": "Use certified seed, apply preventive fungicide, destroy infected plants immediately.",
    },
    "Potato___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; hill soil around stems and watch for pests.",
    },
    # ---- Raspberry ----
    "Raspberry___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; prune old canes and ensure good drainage.",
    },
    # ---- Soybean ----
    "Soybean___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; rotate crops and maintain soil fertility.",
    },
    # ---- Squash ----
    "Squash___Powdery_mildew": {
        "severity": "Moderate",
        "description": "Fungal disease producing white powdery patches on leaves; can reduce yield.",
        "advice": "Improve air circulation, apply sulfur or potassium bicarbonate, plant resistant varieties.",
    },
    # ---- Strawberry ----
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "description": "Fungal disease causing irregular purple to brown spots that merge, giving a scorched look.",
        "advice": "Remove affected leaves, improve air circulation, apply fungicide if severe.",
    },
    "Strawberry___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; renew beds every few years and mulch.",
    },
    # ---- Tomato ----
    "Tomato___Bacterial_spot": {
        "severity": "Moderate",
        "description": "Bacterial disease causing small dark water-soaked spots on leaves and fruit.",
        "advice": "Use disease-free seed, rotate crops, apply copper sprays if detected.",
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark spots with concentric rings on older leaves; common in humid conditions.",
        "advice": "Remove lower leaves, mulch to prevent splash, apply fungicide regularly.",
    },
    "Tomato___Late_blight": {
        "severity": "Severe",
        "description": "Devastating disease causing large greasy gray-green patches; spreads rapidly in cool, wet weather.",
        "advice": "Apply preventive fungicide, destroy infected plants, avoid overhead watering.",
    },
    "Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "description": "Fungal disease causing pale yellow spots on upper leaves and olive-green mold underneath.",
        "advice": "Improve ventilation (esp. in greenhouses), reduce humidity, apply fungicide.",
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing small circular spots with dark borders and light centers on lower leaves.",
        "advice": "Remove affected leaves, mulch, avoid overhead watering, apply fungicide.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate",
        "description": "Tiny pests causing stippled, yellowing leaves; fine webbing may be visible.",
        "advice": "Spray with water to dislodge, use insecticidal soap or neem oil, encourage predators.",
    },
    "Tomato___Target_Spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing brown spots with concentric rings (target-like), similar to Early blight.",
        "advice": "Improve air circulation, remove affected leaves, apply fungicide.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Severe",
        "description": "Viral disease spread by whiteflies; causes upward curling and yellowing of leaves, stunted growth.",
        "advice": "Control whitefly vectors, remove infected plants, use resistant varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "Moderate",
        "description": "Viral disease causing mottled light/dark green pattern on leaves; spread by contact.",
        "advice": "Disinfect tools, remove infected plants, avoid tobacco near plants.",
    },
    "Tomato___healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "advice": "Keep monitoring; water consistently and stake plants for airflow.",
    },
}


# ============================================================================
# Helper functions
# ============================================================================
def parse_class(raw: str) -> tuple[str, str, bool]:
    """Split 'Tomato___Early_blight' into (plant, condition, is_healthy)."""
    if "___" in raw:
        plant, condition = raw.split("___", 1)
    else:
        plant, condition = raw, "unknown"
    plant = plant.replace("_", " ").strip().rstrip(",")
    condition_clean = condition.replace("_", " ").strip()
    is_healthy = condition.lower() == "healthy"
    return plant, condition_clean, is_healthy


def predict(model, idx_to_class, device, image: Image.Image, k: int = 5):
    """Run inference; return list of (class_name, probability)."""
    tf = build_transform()
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu()
    top_probs, top_idx = torch.topk(probs, k=min(k, len(idx_to_class)))
    return [(idx_to_class[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]


def auto_crop_leaf(pil_image: Image.Image, margin: int = 15) -> Image.Image:
    """
    Detect the main leaf in an image and crop to its bounding box.
    Uses a two-stage approach:
      1. HSV color masking to find likely leaf pixels (green/brown)
      2. GrabCut refinement on that region

    Falls back to the original image if detection fails.
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    # --- Stage 1: HSV color mask for leaf-like pixels ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Green range (healthy leaves)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Brown/yellow range (diseased parts)
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([25, 255, 220])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    leaf_mask = cv2.bitwise_or(mask_green, mask_brown)

    # Clean up noise
    kernel = np.ones((7, 7), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    # Find largest connected leaf region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, 8)
    if num_labels <= 1:
        return pil_image  # no leaf found

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x, y, bw, bh = (stats[largest, cv2.CC_STAT_LEFT],
                    stats[largest, cv2.CC_STAT_TOP],
                    stats[largest, cv2.CC_STAT_WIDTH],
                    stats[largest, cv2.CC_STAT_HEIGHT])

    # Reject tiny detections
    if bw * bh < 0.05 * w * h:
        return pil_image

    # --- Stage 2: GrabCut refinement on the bounding box ---
    small_scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
    small = cv2.resize(img, (int(w * small_scale), int(h * small_scale)))
    sh, sw = small.shape[:2]

    rect = (max(0, int(x * small_scale) - 5),
            max(0, int(y * small_scale) - 5),
            min(sw - 1, int(bw * small_scale) + 10),
            min(sh - 1, int(bh * small_scale) + 10))

    gc_mask = np.zeros((sh, sw), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n2 > 1:
            a2 = st2[1:, cv2.CC_STAT_AREA]
            i2 = 1 + int(np.argmax(a2))
            x2, y2, bw2, bh2 = (st2[i2, cv2.CC_STAT_LEFT],
                                st2[i2, cv2.CC_STAT_TOP],
                                st2[i2, cv2.CC_STAT_WIDTH],
                                st2[i2, cv2.CC_STAT_HEIGHT])
            if bw2 * bh2 > 0.05 * sw * sh:
                inv = 1.0 / small_scale
                x = int(x2 * inv)
                y = int(y2 * inv)
                bw = int(bw2 * inv)
                bh = int(bh2 * inv)
    except cv2.error:
        pass  # keep color-based box

    # Crop with margin, clamped to image bounds
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + bw + margin)
    y1 = min(h, y + bh + margin)

    return pil_image.crop((x0, y0, x1, y1))


def generate_gradcam(model, device, image: Image.Image, target_class: int):
    """
    Pure-PyTorch Grad-CAM (no external library).

    Computes the gradient of the target class score w.r.t. the last
    convolutional feature map, then produces a heatmap overlay on the
    input image.

    Returns an RGB numpy array (uint8) with the heatmap overlaid.
    """
    model.eval()

    # Hook to capture the last conv feature map + its gradient
    features = {}
    gradients = {}

    def forward_hook(module, input, output):
        features["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Last conv block of MobileNetV2
    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        tf = build_transform()
        input_tensor = tf(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        model.zero_grad()
        logits = model(input_tensor)

        # Gradient of the target class score
        score = logits[0, target_class]
        score.backward()

        # feature map: (1, C, H, W)  -> gradient: (1, C, H, W)
        fmap = features["value"][0]           # (C, H, W)
        grad = gradients["value"][0]          # (C, H, W)

        # Global-average-pool the gradients over spatial dims -> weights (C,)
        weights = grad.mean(dim=(1, 2))

        # Weighted sum of feature maps
        cam = torch.zeros(fmap.shape[1:], dtype=torch.float32, device=fmap.device)
        for i, w in enumerate(weights):
            cam += w * fmap[i]

        # ReLU (only positive contributions)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy()

        # Resize heatmap to 224x224
        cam_img = cv2.resize(cam_np, (224, 224))

        # Apply JET colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_img), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32)
        overlay = heatmap.astype(np.float32) * 0.4 + rgb_img * 0.6
        overlay = np.uint8(np.clip(overlay, 0, 255))

        return overlay
    finally:
        fh.remove()
        bh.remove()
def confidence_gauge(prob: float, is_healthy: bool) -> str:
    """Return HTML for a circular confidence gauge (SVG)."""
    pct = prob * 100
    color = "#4ade80" if is_healthy else "#f87171"
    # Circle circumference = 2*pi*r = 2*pi*45 ≈ 282.74
    circumference = 282.74
    offset = circumference * (1 - prob)
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;">
      <svg width="140" height="140" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="45" stroke="rgba(74,222,128,0.15)"
                stroke-width="10" fill="none"/>
        <circle cx="60" cy="60" r="45" stroke="{color}"
                stroke-width="10" fill="none" stroke-linecap="round"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                transform="rotate(-90 60 60)"
                style="transition: stroke-dashoffset 0.6s ease;"/>
        <text x="60" y="58" text-anchor="middle" fill="#f0fdf4"
              font-size="22" font-weight="700">{pct:.1f}%</text>
        <text x="60" y="76" text-anchor="middle" fill="#86efac"
              font-size="10" font-weight="600">CONFIDENCE</text>
      </svg>
    </div>
    """


# ============================================================================
# Load model and metrics
# ============================================================================
model, idx_to_class, device, ckpt = load_model()
metrics = load_metrics()

# ============================================================================
# Header — Hero section
# ============================================================================
st.markdown("""
<div class="pv-hero">
    <h1>🌿 PlantVision AI</h1>
    <p>From Plant Images to Agricultural Intelligence.</p>
    <p style="font-size:0.95rem;opacity:0.8;margin-top:0.5rem;">
        Upload a leaf photo — identify the plant and assess its health in seconds.
    </p>
    <span class="pv-badge">🌱 Classifier Preview · Phase 10</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar — Model info
# ============================================================================
with st.sidebar:
    st.markdown("### 🌿 PlantVision AI")
    st.caption("Agricultural Intelligence Platform")
    st.divider()

    st.markdown("#### 🧠 Model Status")
    st.markdown(f"""
    <div class="pv-metric-card">
        <div class="pv-metric-label">Architecture</div>
        <div class="pv-metric-value" style="font-size:1.1rem;">MobileNetV2</div>
    </div>
    <div class="pv-metric-card">
        <div class="pv-metric-label">Device</div>
        <div class="pv-metric-value" style="font-size:1.1rem;">{device}</div>
    </div>
    <div class="pv-metric-card">
        <div class="pv-metric-label">Classes</div>
        <div class="pv-metric-value">{len(idx_to_class)}</div>
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        st.markdown("#### 📊 Test Performance")
        st.markdown(f"""
        <div class="pv-metric-card">
            <div class="pv-metric-label">Accuracy</div>
            <div class="pv-metric-value">{metrics['accuracy']:.2%}</div>
        </div>
        <div class="pv-metric-card">
            <div class="pv-metric-label">Top-5 Accuracy</div>
            <div class="pv-metric-value">{metrics['top5_accuracy']:.2%}</div>
        </div>
        <div class="pv-metric-card">
            <div class="pv-metric-label">F1 (Macro)</div>
            <div class="pv-metric-value">{metrics['f1_macro']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Evaluated on {metrics['num_test_images']:,} held-out images")

    st.divider()
    st.caption(
        "**Preview build.** Pest detection, severity estimation, health score, "
        "and RAG recommendations are planned for later phases."
    )

# ============================================================================
# Tabs
# ============================================================================
tab_predict, tab_info, tab_analytics = st.tabs(["🔍 Predict", "📊 Model Info", "📈 Analytics"])
# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    # --- Upload section ---
    col_upload_left, col_upload_right = st.columns([2, 1])

    with col_upload_left:
        uploaded = st.file_uploader(
            "📸 Upload a plant leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="One leaf, plain background, good lighting = best results.",
        )

    with col_upload_right:
        auto_crop = st.toggle(
            "✂️ Auto-crop leaf",
            value=True,
            help="Automatically isolate the main leaf before prediction.",
        )
        st.caption("Recommended for real-world photos.")

    # --- Tips expander ---
    with st.expander("📖 Tips for best accuracy", expanded=False):
        st.markdown("""
        The model was trained on **PlantVillage-style** images: a single leaf,
        plain background, even lighting.

        ✅ **DO:**
        - Photograph **one leaf only** — fill the frame
        - Use a **plain background** (white paper, cloth)
        - Use **natural daylight**
        - Hold camera **parallel** to the leaf (top-down)

        ❌ **DON'T:**
        - Photograph the whole plant or a full branch
        - Use busy backgrounds
        - Use flash or harsh shadows
        - Zoom out — get close
        """)

    # --- Result section ---
    if uploaded is None:
        st.info("👆 Upload an image above to see a prediction.")
    else:
        original = Image.open(uploaded).convert("RGB")

        if auto_crop:
            with st.spinner("Detecting leaf..."):
                processed = auto_crop_leaf(original)
            crop_applied = processed.size != original.size
        else:
            processed = original
            crop_applied = False

        # Low-res warning
        w, h = processed.size
        if min(w, h) < 200:
            st.warning(f"⚠️ Image is small ({w}×{h}). Predictions may be unreliable.")

        # --- Run prediction ---
        with st.spinner("Analyzing leaf..."):
            results = predict(model, idx_to_class, device, processed)

        top_class, top_prob = results[0]
        plant, condition, is_healthy = parse_class(top_class)

        # --- Image column ---
        col_img, col_result = st.columns([1, 1.2])

        with col_img:
            if auto_crop:
                st.image(original, caption="Original", use_container_width=True)
                st.image(processed, caption="After auto-crop", use_container_width=True)
                if not crop_applied:
                    st.caption("_(No leaf detected — using original)_")
            else:
                st.image(processed, caption="Uploaded image", use_container_width=True)

        # --- Result column ---
        with col_result:
            result_class = "pv-result-healthy" if is_healthy else "pv-result-diseased"
            icon = "✅" if is_healthy else "⚠️"
            condition_display = "Healthy" if is_healthy else condition

            st.markdown(f"""
            <div class="pv-result {result_class}">
                <div class="pv-result-icon">{icon}</div>
                <p class="pv-result-plant">{plant}</p>
                <p class="pv-result-condition">{condition_display}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Confidence gauge
            st.markdown(confidence_gauge(top_prob, is_healthy), unsafe_allow_html=True)

            # Low confidence warning
            if top_prob < 0.80:
                st.warning(
                    f"⚠️ **Low confidence ({top_prob:.1%}).** The image may differ "
                    f"from the training data. Try auto-crop or retake the photo."
                )
            elif top_prob < 0.95:
                st.info(
                    f"ℹ️ Moderate confidence ({top_prob:.1%}). "
                    f"Consider retaking for a more reliable result."
                )

        # --- Disease info ---
        info = DISEASE_INFO.get(top_class)
        if info:
            st.markdown("---")
            st.markdown("### 📋 About this result")
            col_a, col_b = st.columns(2)

            with col_a:
                sev_color = {
                    "None": "#4ade80",
                    "Moderate": "#fbbf24",
                    "Severe": "#f87171",
                }.get(info["severity"], "#94a3b8")
                st.markdown(f"""
                <div class="pv-info-box">
                    <h4>🩺 Condition</h4>
                    <p><strong style="color:{sev_color};">Severity: {info['severity']}</strong></p>
                    <p style="margin-top:0.5rem;">{info['description']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class="pv-info-box">
                    <h4>💡 Care advice</h4>
                    <p>{info['advice']}</p>
                </div>
                """, unsafe_allow_html=True)

        # --- Top-5 candidates ---
        st.markdown("---")
        st.markdown("### 🎯 Top 5 candidates")

        for i, (raw_class, prob) in enumerate(results):
            p, c, healthy = parse_class(raw_class)
            label = f"🌱 {p} — Healthy" if healthy else f"🦠 {p} — {c}"
            rank_class = "pv-rank-item first" if i == 0 else "pv-rank-item"
            st.markdown(f"""
            <div class="{rank_class}">
                <span class="pv-rank-name">#{i+1} · {label}</span>
                <span class="pv-rank-prob">{prob:.2%}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- Grad-CAM attention map ---
        st.markdown("---")
        st.markdown("### 🔍 Model Attention (Grad-CAM)")
        st.caption(
            "Where the model looked when making its top prediction. "
            "Red = high attention, blue = low attention."
        )

        with st.spinner("Generating attention map..."):
            try:
                top_idx = [k for k, v in idx_to_class.items() if v == top_class][0]
                heatmap = generate_gradcam(model, device, processed, target_class=top_idx)
                st.image(
                    heatmap,
                    caption=f"Attention for: {plant} — {condition}",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Could not generate Grad-CAM: {e}")

        # --- Disclaimer ---
        st.markdown("---")
        st.caption(
            "⚠️ This is a raw classifier output for a single image — not a full "
            "plant health report. Confirm with a professional before any treatment."
        )

# ---------------------------------------------------------------------------
# Model Info tab
# ---------------------------------------------------------------------------
with tab_info:
    st.markdown("## 🧠 Model Details")
    st.markdown("""
    **Architecture:** MobileNetV2 (transfer learning from ImageNet)

    **Training data:** PlantVillage — 38 classes across 14 crops

    **Training setup:**
    - Backbone frozen (head-only training)
    - Adam optimizer, lr=1e-3
    - 20 epochs max, early best-checkpoint saving
    - WeightedRandomSampler for class imbalance
    - Mixed-precision (AMP) on GPU
    """)

    if metrics:
        st.markdown("---")
        st.markdown("### 📊 Test Set Metrics (held-out)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Top-5 Accuracy", f"{metrics['top5_accuracy']:.2%}")
        m3.metric("F1 (macro)", f"{metrics['f1_macro']:.2%}")
        m4.metric("F1 (weighted)", f"{metrics['f1_weighted']:.2%}")

        st.caption(
            f"Evaluated on **{metrics['num_test_images']:,}** images never seen "
            f"during training or validation. All numbers come from "
            f"`reports/plant_classifier_metrics.json` — none are hand-written."
        )

        # --- Worst 5 classes ---
        st.markdown("---")
        st.markdown("### ⚠️ Weakest 5 Classes")
        st.caption("These classes have the lowest per-class F1 — useful for future improvement.")

        for row in metrics.get("worst_5_classes", []):
            p, c, _ = parse_class(row["class"])
            f1 = row["f1"]
            st.markdown(f"""
            <div class="pv-rank-item">
                <span class="pv-rank-name">{p} — {c}</span>
                <span class="pv-rank-prob" style="color:#fbbf24;">F1 = {f1:.2%}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- Confusion matrix ---
        if CONFUSION_MATRIX_PATH.exists():
            st.markdown("---")
            st.markdown("### 🔀 Confusion Matrix")
            st.caption("Full 38×38 confusion matrix on the test set. Darker = more predictions.")
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)

        # --- All classes ---
        st.markdown("---")
        st.markdown("### 🌾 All 38 Recognized Classes")
        crops = sorted({parse_class(c)[0] for c in idx_to_class.values()})
        st.markdown(f"**{len(crops)} crops covered:**")
        st.markdown(
            " · ".join([f"`{c}`" for c in crops])
        )
    else:
        st.warning("No metrics file found. Run `python training/evaluate.py` first.")

    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - Works best on **clear, single-leaf photos** with a plain background
    - Real-world field photos may reduce accuracy
    - Only 38 classes covered — unfamiliar plants will be misclassified
    - Pest detection and severity estimation are separate modules (planned)
    - Not a substitute for professional agricultural diagnosis
    """)

# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------
with tab_analytics:
    st.markdown("## 📈 Analytics Dashboard")
    st.caption("Real, run-produced statistics from the dataset and the trained model.")

    split_counts = count_dataset_images()
    per_class_df = count_per_class()
    log_df = load_training_log()

    # ===== Overview cards =====
    st.markdown("### 🎯 Overview")
    total_images = sum(split_counts.values()) if split_counts else 0
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Total images", f"{total_images:,}")
    o2.metric("Classes", len(idx_to_class))
    if metrics:
        o3.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
        o4.metric("F1 (macro)", f"{metrics['f1_macro']:.2%}")

    # ===== Dataset split =====
    if split_counts:
        st.markdown("---")
        st.markdown("### 🗂️ Dataset Split")
        split_df = pd.DataFrame({
            "Split": list(split_counts.keys()),
            "Images": list(split_counts.values()),
        })
        fig_split = px.bar(
            split_df, x="Split", y="Images", color="Split",
            color_discrete_map={
                "train": "#4ade80", "validation": "#fbbf24", "test": "#60a5fa",
            },
            text="Images",
        )
        fig_split.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=False, height=350,
        )
        fig_split.update_traces(textposition="outside")
        st.plotly_chart(fig_split, use_container_width=True)

    # ===== Class distribution =====
    if per_class_df is not None and not per_class_df.empty:
        st.markdown("---")
        st.markdown("### 📊 Top 15 Classes (by training images)")

        top15 = per_class_df.head(15).copy()
        fig_top = px.bar(
            top15, x="count", y="class", orientation="h",
            color="count", color_continuous_scale="Greens",
        )
        fig_top.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=500,
            xaxis_title="Training images", yaxis_title="",
            coloraxis_showscale=False,
        )
        fig_top.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_top, use_container_width=True)

    # ===== Per-class F1 =====
    if metrics and metrics.get("per_class_f1"):
        st.markdown("---")
        st.markdown("### 🎯 Per-Class F1 Score")

        f1_df = pd.DataFrame(
            [{"class": k, "f1": v} for k, v in metrics["per_class_f1"].items()]
        ).sort_values("f1", ascending=True)

        fig_f1 = px.bar(
            f1_df, x="f1", y="class", orientation="h",
            color="f1",
            color_continuous_scale=["#f87171", "#fbbf24", "#4ade80"],
            range_color=(0, 1),
        )
        fig_f1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=950,
            xaxis_title="F1 score", yaxis_title="",
            coloraxis_showscale=False,
        )
        fig_f1.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_f1, use_container_width=True)

    # ===== Training curves =====
    if log_df is not None and not log_df.empty:
        st.markdown("---")
        st.markdown("### 📉 Training History")

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=log_df["epoch"], y=log_df["train_loss"],
            mode="lines+markers", name="Train loss",
            line=dict(color="#60a5fa", width=3),
        ))
        fig_loss.add_trace(go.Scatter(
            x=log_df["epoch"], y=log_df["val_loss"],
            mode="lines+markers", name="Val loss",
            line=dict(color="#f87171", width=3),
        ))
        fig_loss.update_layout(
            title="Loss over epochs",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=400,
            xaxis_title="Epoch", yaxis_title="Loss",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=log_df["epoch"], y=log_df["train_acc"],
            mode="lines+markers", name="Train accuracy",
            line=dict(color="#4ade80", width=3),
        ))
        fig_acc.add_trace(go.Scatter(
            x=log_df["epoch"], y=log_df["val_acc"],
            mode="lines+markers", name="Val accuracy",
            line=dict(color="#fbbf24", width=3),
        ))
        fig_acc.update_layout(
            title="Accuracy over epochs",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=400,
            xaxis_title="Epoch", yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        best_epoch_row = log_df.loc[log_df["val_acc"].idxmax()]
        st.info(
            f"🏆 Best epoch: {int(best_epoch_row['epoch'])} "
            f"with validation accuracy {best_epoch_row['val_acc']:.2%}"
        )

    st.markdown("---")
    st.caption("📌 All statistics are computed live from real artifacts in `reports/`.")


# ============================================================================
# Footer
# ============================================================================
st.markdown("""
<div class="pv-footer">
    <p><strong>PlantVision AI</strong> · From Plant Images to Agricultural Intelligence</p>
    <p>Preview build · Classifier only · For educational use</p>
    <p style="font-size:0.78rem;opacity:0.7;">
        Results should not replace professional agricultural diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)