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
import io
import json
import math
import random
import sqlite3
import sys
from datetime import datetime
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
HISTORY_DB_PATH = ROOT / "data" / "plantvision_history.db"
SAMPLES_DIR = ROOT / "data" / "test"

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
# Extended visual layer — aurora background, glassmorphism, chips, animations
# ============================================================================
PRO_CSS = """
<style>
    /* ---- Animated aurora backdrop ---- */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(38rem 38rem at 12% 8%,  rgba(74, 222, 128, 0.16), transparent 60%),
            radial-gradient(32rem 32rem at 88% 14%, rgba(34, 197, 94, 0.13),  transparent 62%),
            radial-gradient(40rem 40rem at 50% 96%, rgba(16, 185, 129, 0.12), transparent 65%);
        animation: pvDrift 22s ease-in-out infinite alternate;
    }
    .stApp > div { position: relative; z-index: 1; }

    @keyframes pvDrift {
        0%   { transform: translate3d(0, 0, 0) scale(1); }
        50%  { transform: translate3d(-1.5%, 1.5%, 0) scale(1.05); }
        100% { transform: translate3d(1.5%, -1%, 0) scale(1.02); }
    }

    /* ---- Hero upgrade: leaf-vein texture + shimmer ---- */
    .pv-hero {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(120% 140% at 0% 0%, rgba(74,222,128,0.22) 0%, transparent 55%),
            linear-gradient(135deg, #16412a 0%, #2d6a4f 55%, #1b5e3a 100%) !important;
    }
    .pv-hero::after {
        content: "";
        position: absolute;
        top: -60%;
        left: -30%;
        width: 60%;
        height: 220%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
        transform: rotate(18deg);
        animation: pvShimmer 7s ease-in-out infinite;
    }
    @keyframes pvShimmer {
        0%   { left: -40%; opacity: 0; }
        35%  { opacity: 1; }
        100% { left: 130%; opacity: 0; }
    }
    .pv-hero-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.35rem;
        position: relative;
        z-index: 2;
    }
    .pv-hero-top {
        display: flex;
        align-items: center;
        gap: 1.4rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 2;
    }
    .pv-hero-copy { flex: 1; min-width: 260px; }
    .pv-hero-eyebrow {
        color: #6ee7b7;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
        opacity: 0.9;
    }
    .pv-hero h1 {
        background: linear-gradient(95deg, #ffffff 0%, #d1fae5 45%, #6ee7b7 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        line-height: 1.05;
    }
    .pv-hero-sub {
        font-size: 0.95rem !important;
        opacity: 0.75;
        margin-top: 0.35rem !important;
    }
    @media (max-width: 640px) {
        .pv-hero h1 { font-size: 2.1rem !important; }
    }

    /* ---- Glass cards ---- */
    .pv-glass {
        background: rgba(20, 45, 32, 0.45);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(134, 239, 172, 0.18);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }
    .pv-glass h4 {
        color: #86efac;
        margin: 0 0 0.6rem 0;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .pv-metric-card {
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .pv-metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(74, 222, 128, 0.5);
    }

    /* ---- Chips / pills ---- */
    .pv-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(134, 239, 172, 0.28);
        background: rgba(74, 222, 128, 0.12);
        color: #bbf7d0;
        margin: 0 0.35rem 0.35rem 0;
    }
    .pv-chip-warn   { background: rgba(251, 191, 36, 0.14); border-color: rgba(251,191,36,0.35); color: #fde68a; }
    .pv-chip-danger { background: rgba(248, 113, 113, 0.14); border-color: rgba(248,113,113,0.35); color: #fecaca; }
    .pv-chip-info   { background: rgba(96, 165, 250, 0.14); border-color: rgba(96,165,250,0.35); color: #bfdbfe; }

    /* ---- Severity / progress bars ---- */
    .pv-bar-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
        margin: 0.45rem 0 0.2rem 0;
    }
    .pv-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.9s cubic-bezier(.22,.9,.25,1);
    }

    /* ---- History timeline ---- */
    .pv-timeline-item {
        display: flex;
        gap: 1rem;
        align-items: center;
        background: rgba(26, 77, 46, 0.28);
        border: 1px solid rgba(74, 222, 128, 0.15);
        border-radius: 14px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.55rem;
    }
    .pv-timeline-item img {
        width: 56px; height: 56px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid rgba(134, 239, 172, 0.3);
    }
    .pv-timeline-main  { flex: 1; }
    .pv-timeline-title { color: #f0fdf4; font-weight: 700; font-size: 0.95rem; }
    .pv-timeline-sub   { color: #94a3b8; font-size: 0.78rem; margin-top: 0.15rem; }
    .pv-timeline-score { font-size: 1.25rem; font-weight: 800; }

    /* ---- Knowledge cards ---- */
    .pv-kb-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(74, 222, 128, 0.16);
        border-left: 4px solid #4ade80;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
    }
    .pv-kb-card h5 { color: #f0fdf4; margin: 0 0 0.35rem 0; font-size: 1.02rem; }
    .pv-kb-card p  { color: #cbd5e1; font-size: 0.9rem; margin: 0.3rem 0; line-height: 1.55; }

    /* ---- Scanning pulse (while analysing) ---- */
    .pv-scanning {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
    }
    .pv-scanning::after {
        content: "";
        position: absolute; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, #4ade80, transparent);
        animation: pvScan 1.6s linear infinite;
    }
    @keyframes pvScan { 0% { top: 0; } 100% { top: 100%; } }

    /* ---- Streamlit polish ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1712 0%, #12241a 100%);
        border-right: 1px solid rgba(74, 222, 128, 0.12);
    }
    div[data-testid="stMetricValue"] { color: #f0fdf4; }
    div[data-testid="stMetricLabel"] { color: #86efac; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
    ::-webkit-scrollbar-thumb {
        background: rgba(74, 222, 128, 0.28);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(74, 222, 128, 0.45); }
</style>
"""

RTL_CSS = """
<style>
    .stApp, .stMarkdown, .pv-glass, .pv-info-box, .pv-kb-card { direction: rtl; text-align: right; }
    .pv-rank-item { direction: rtl; border-left: none; border-right: 4px solid #4ade80; }
    .pv-rank-item.first { border-right-color: #22c55e; }
    .pv-info-box, .pv-kb-card { border-left: none; border-right: 4px solid #4ade80; }
    .pv-timeline-item { direction: rtl; }
    code, pre, .stDataFrame { direction: ltr; text-align: left; }
</style>
"""

st.markdown(PRO_CSS, unsafe_allow_html=True)


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
# Bilingual translations (EN / AR-EG)
# ============================================================================
TRANSLATIONS = {
    "en": {
        # Header
        "hero_tagline": "From Plant Images to Agricultural Intelligence.",
        "hero_subtitle": "Upload a leaf photo — identify the plant and assess its health in seconds.",
        "hero_badge": "🌱 Classifier Preview · Phase 10",

        # Sidebar
        "sidebar_title": "PlantVision AI",
        "sidebar_subtitle": "Agricultural Intelligence Platform",
        "sidebar_model": "Model Status",
        "sidebar_arch": "Architecture",
        "sidebar_device": "Device",
        "sidebar_classes": "Classes",
        "sidebar_perf": "Test Performance",
        "sidebar_acc": "Accuracy",
        "sidebar_top5": "Top-5 Accuracy",
        "sidebar_f1": "F1 (Macro)",
        "sidebar_evaluated": "Evaluated on",
        "sidebar_evaluated_imgs": "held-out images",
        "sidebar_preview_note": "**Preview build.** Pest detection, severity estimation, health score, and RAG recommendations are planned for later phases.",

        # Tabs
        "tab_predict": "🔍 Predict",
        "tab_info": "📊 Model Info",
        "tab_analytics": "📈 Analytics",

        # Predict tab
        "upload_label": "📸 Upload a plant leaf image (JPG / PNG)",
        "upload_help": "One leaf, plain background, good lighting = best results.",
        "autocrop_label": "✂️ Auto-crop leaf",
        "autocrop_help": "Automatically isolate the main leaf before prediction.",
        "autocrop_caption": "Recommended for real-world photos.",
        "tips_title": "📖 Tips for best accuracy",
        "no_image": "👆 Upload an image above to see a prediction.",
        "analyzing": "Analyzing leaf...",
        "detecting": "Detecting leaf...",
        "original": "Original",
        "after_crop": "After auto-crop",
        "no_leaf": "_(No leaf detected — using original)_",
        "uploaded_image": "Uploaded image",
        "healthy": "Healthy",
        "low_conf": "⚠️ **Low confidence ({pct}).** The image may differ from the training data. Try auto-crop or retake the photo.",
        "mod_conf": "ℹ️ Moderate confidence ({pct}). Consider retaking for a more reliable result.",
        "about_result": "📋 About this result",
        "condition": "🩺 Condition",
        "severity": "Severity",
        "care_advice": "💡 Care advice",
        "top5_title": "🎯 Top 5 candidates",
        "gradcam_title": "🔍 Model Attention (Grad-CAM)",
        "gradcam_caption": "Where the model looked when making its top prediction. Red = high attention, blue = low attention.",
        "gradcam_spinner": "Generating attention map...",
        "gradcam_attn": "Attention for",
        "gradcam_fail": "Could not generate Grad-CAM",
        "disclaimer": "⚠️ This is a raw classifier output for a single image — not a full plant health report. Confirm with a professional before any treatment.",
        "low_res": "⚠️ Image is small ({w}×{h}). Predictions may be unreliable.",

        # Health score
        "health_score": "Health Score",
        "hs_healthy": "Healthy",
        "hs_moderate": "Moderate",
        "hs_critical": "Critical",
        "hs_desc_healthy": "The leaf looks healthy. Keep monitoring.",
        "hs_desc_moderate": "Signs of disease detected. Take action soon.",
        "hs_desc_critical": "Serious disease. Treat immediately.",

        # Model Info tab
        "info_title": "🧠 Model Details",
        "info_metrics": "📊 Test Set Metrics (held-out)",
        "info_weakest": "⚠️ Weakest 5 Classes",
        "info_weakest_cap": "These classes have the lowest per-class F1 — useful for future improvement.",
        "info_confusion": "🔀 Confusion Matrix",
        "info_confusion_cap": "Full 38×38 confusion matrix on the test set. Darker = more predictions.",
        "info_classes": "🌾 All 38 Recognized Classes",
        "info_crops_covered": "crops covered",
        "info_limitations": "⚠️ Limitations",
        "info_no_metrics": "No metrics file found. Run `python training/evaluate.py` first.",

        # Analytics tab
        "an_title": "## 📈 Analytics Dashboard",
        "an_caption": "Real, run-produced statistics from the dataset and the trained model.",
        "an_overview": "### 🎯 Overview",
        "an_total_images": "Total images",
        "an_classes": "Classes",
        "an_test_acc": "Test accuracy",
        "an_f1": "F1 (macro)",
        "an_split": "### 🗂️ Dataset Split",
        "an_top15": "### 📊 Top 15 Classes (by training images)",
        "an_per_f1": "### 🎯 Per-Class F1 Score",
        "an_training": "### 📉 Training History",
        "an_footer": "📌 All statistics are computed live from real artifacts in `reports/`.",

        # Footer
        "footer_line1": "PlantVision AI · From Plant Images to Agricultural Intelligence",
        "footer_line2": "Preview build · Classifier only · For educational use",
        "footer_line3": "Results should not replace professional agricultural diagnosis.",

        # Language toggle
        "lang_label": "🌐 Language",
        "lang_en": "English",
        "lang_ar": "العربية",
    },
    "ar": {
        # Header
        "hero_tagline": "من صور النباتات إلى الذكاء الزراعي.",
        "hero_subtitle": "ارفع صورة ورقة نبتة — واعرف نوعها وحالتها في ثواني.",
        "hero_badge": "🌱 نسخة تجريبية · المرحلة 10",

        # Sidebar
        "sidebar_title": "PlantVision AI",
        "sidebar_subtitle": "منصة الذكاء الزراعي",
        "sidebar_model": "حالة الموديل",
        "sidebar_arch": "المعمارية",
        "sidebar_device": "الجهاز",
        "sidebar_classes": "عدد الفئات",
        "sidebar_perf": "أداء الموديل",
        "sidebar_acc": "الدقة",
        "sidebar_top5": "دقة أفضل 5",
        "sidebar_f1": "مقياس F1",
        "sidebar_evaluated": "تم التقييم على",
        "sidebar_evaluated_imgs": "صورة اختبار",
        "sidebar_preview_note": "**نسخة تجريبية.** كشف الحشرات وتقدير الشدة ومؤشر الصحة والتوصيات الذكية — كل دي مراحل جاية.",

        # Tabs
        "tab_predict": "🔍 التحليل",
        "tab_info": "📊 معلومات الموديل",
        "tab_analytics": "📈 الإحصائيات",

        # Predict tab
        "upload_label": "📸 ارفع صورة ورقة نبتة (JPG / PNG)",
        "upload_help": "ورقة واحدة، خلفية بسيطة، إضاءة كويسة = أفضل نتيجة.",
        "autocrop_label": "✂️ قص الورقة تلقائيًا",
        "autocrop_help": "الموديل بيعزل الورقة الأساسية لوحده قبل التحليل.",
        "autocrop_caption": "منصوح بيها للصور الواقعية.",
        "tips_title": "📖 نصايح لأفضل نتيجة",
        "no_image": "👆 ارفع صورة فوق الأول عشان تشوف النتيجة.",
        "analyzing": "جاري التحليل...",
        "detecting": "جاري كشف الورقة...",
        "original": "الصورة الأصلية",
        "after_crop": "بعد القص",
        "no_leaf": "_(مفيش ورقة اتكشفت — هنستخدم الصورة الأصلية)_",
        "uploaded_image": "الصورة المرفوعة",
        "healthy": "سليمة",
        "low_conf": "⚠️ **الثقة قليلة ({pct}).** الصورة دي مختلفة عن اللي الموديل اتدرب عليه. جرّب تقص الورقة أو صوّرها تاني.",
        "mod_conf": "ℹ️ الثقة متوسطة ({pct}). جرّب تصوّر الورقة تاني لنتيجة أدق.",
        "about_result": "📋 عن النتيجة دي",
        "condition": "🩺 الحالة",
        "severity": "الشدة",
        "care_advice": "💡 نصايح للعلاج",
        "top5_title": "🎯 أفضل 5 احتمالات",
        "gradcam_title": "🔍 اللي الموديل بص عليه (Grad-CAM)",
        "gradcam_caption": "المناطق اللي الموديل ركّز عليها لما أخد القرار. الأحمر = تركيز عالي، الأزرق = تركيز ضعيف.",
        "gradcam_spinner": "جاري رسم الخريطة...",
        "gradcam_attn": "تركيز على",
        "gradcam_fail": "مش قادرين نولّد Grad-CAM",
        "disclaimer": "⚠️ ده تحليل مبدئي لصورة واحدة — مش تقرير كامل عن صحة النبتة. اتأكد مع خبير قبل أي علاج.",
        "low_res": "⚠️ الصورة صغيرة ({w}×{h}). النتيجة مش دقيقة.",

        # Health score
        "health_score": "مؤشر الصحة",
        "hs_healthy": "سليمة",
        "hs_moderate": "متوسطة",
        "hs_critical": "خطيرة",
        "hs_desc_healthy": "الورقة شكلها سليمة. خلّي عينك عليها.",
        "hs_desc_moderate": "فيه علامات مرض. اتحرك بسرعة.",
        "hs_desc_critical": "المرض خطير. لازم علاج فوري.",

        # Model Info tab
        "info_title": "🧠 تفاصيل الموديل",
        "info_metrics": "📊 نتائج الاختبار",
        "info_weakest": "⚠️ أضعف 5 فئات",
        "info_weakest_cap": "دي الفئات اللي الموديل أضعف فيها — مفيدة لتطوير المشروع بعدين.",
        "info_confusion": "🔀 مصفوفة الالتباس",
        "info_confusion_cap": "مصفوفة 38×38 للاختبار. الأغمق = تنبؤات أكتر.",
        "info_classes": "🌾 كل الـ 38 فئة",
        "info_crops_covered": "فصيلة مدعومة",
        "info_limitations": "⚠️ قيود",
        "info_no_metrics": "مفيش ملف نتايج. شغّل `python training/evaluate.py` الأول.",

        # Analytics tab
        "an_title": "## 📈 لوحة الإحصائيات",
        "an_caption": "إحصائيات حقيقية من الداتاسيت والموديل.",
        "an_overview": "### 🎯 نظرة عامة",
        "an_total_images": "إجمالي الصور",
        "an_classes": "عدد الفئات",
        "an_test_acc": "دقة الاختبار",
        "an_f1": "مقياس F1",
        "an_split": "### 🗂️ تقسيم الداتاسيت",
        "an_top15": "### 📊 أفضل 15 فئة (بعدد الصور)",
        "an_per_f1": "### 🎯 مقياس F1 لكل فئة",
        "an_training": "### 📉 تاريخ التدريب",
        "an_footer": "📌 كل الأرقام دي محسوبة مباشرة من ملفات حقيقية في `reports/`.",

        # Footer
        "footer_line1": "PlantVision AI · من صور النباتات إلى الذكاء الزراعي",
        "footer_line2": "نسخة تجريبية · الموديل فقط · للاستخدام التعليمي",
        "footer_line3": "النتايج مش بديل عن التشخيص الزراعي المتخصص.",

        # Language toggle
        "lang_label": "🌐 اللغة",
        "lang_en": "English",
        "lang_ar": "العربية",
    },
}


# Arabic translations for the 38 classes (plant name, condition)
AR_CLASS_NAMES = {
    "Apple___Apple_scab": ("تفاح", "جرب التفاح"),
    "Apple___Black_rot": ("تفاح", "العفن الأسود"),
    "Apple___Cedar_apple_rust": ("تفاح", "صدأ الأرز"),
    "Apple___healthy": ("تفاح", "سليمة"),
    "Blueberry___healthy": ("بلوبيري", "سليمة"),
    "Cherry_(including_sour)___Powdery_mildew": ("كرز", "البياض الدقيقي"),
    "Cherry_(including_sour)___healthy": ("كرز", "سليمة"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("ذرة", "تبقع الأوراق الرمادي"),
    "Corn_(maize)___Common_rust_": ("ذرة", "الصدأ الشائع"),
    "Corn_(maize)___Northern_Leaf_Blight": ("ذرة", "لفحة الأوراق الشمالية"),
    "Corn_(maize)___healthy": ("ذرة", "سليمة"),
    "Grape___Black_rot": ("عنب", "العفن الأسود"),
    "Grape___Esca_(Black_Measles)": ("عنب", "الأسكا (الحصبة السوداء)"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("عنب", "لفحة الأوراق"),
    "Grape___healthy": ("عنب", "سليمة"),
    "Orange___Haunglongbing_(Citrus_greening)": ("برتقال", "التخضير الحمضي"),
    "Peach___Bacterial_spot": ("خوخ", "التبقع البكتيري"),
    "Peach___healthy": ("خوخ", "سليمة"),
    "Pepper,_bell___Bacterial_spot": ("فلفل رومي", "التبقع البكتيري"),
    "Pepper,_bell___healthy": ("فلفل رومي", "سليمة"),
    "Potato___Early_blight": ("بطاطس", "اللفحة المبكرة"),
    "Potato___Late_blight": ("بطاطس", "اللفحة المتأخرة"),
    "Potato___healthy": ("بطاطس", "سليمة"),
    "Raspberry___healthy": ("توت العليق", "سليمة"),
    "Soybean___healthy": ("فول الصويا", "سليمة"),
    "Squash___Powdery_mildew": ("قرع", "البياض الدقيقي"),
    "Strawberry___Leaf_scorch": ("فراولة", "لفحة الأوراق"),
    "Strawberry___healthy": ("فراولة", "سليمة"),
    "Tomato___Bacterial_spot": ("طماطم", "التبقع البكتيري"),
    "Tomato___Early_blight": ("طماطم", "اللفحة المبكرة"),
    "Tomato___Late_blight": ("طماطم", "اللفحة المتأخرة"),
    "Tomato___Leaf_Mold": ("طماطم", "عفن الأوراق"),
    "Tomato___Septoria_leaf_spot": ("طماطم", "تبقع سبتوريا"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("طماطم", "العنكبوت الأحمر"),
    "Tomato___Target_Spot": ("طماطم", "التبقع الهدفي"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("طماطم", "فيروس تجعد الأوراق الأصفر"),
    "Tomato___Tomato_mosaic_virus": ("طماطم", "فيروس الموزاييك"),
    "Tomato___healthy": ("طماطم", "سليمة"),
}


# ----------------------------------------------------------------------------
# Strings for the new modules (merged into TRANSLATIONS below)
# ----------------------------------------------------------------------------
EXTRA_TRANSLATIONS = {
    "en": {
        # Tabs
        "tab_batch": "📦 Batch Scan",
        "tab_history": "🕒 History",
        "tab_kb": "📚 Knowledge Base",

        # Input sources
        "src_title": "Image source",
        "src_upload": "📁 Upload",
        "src_camera": "📷 Camera",
        "src_sample": "🎲 Sample leaf",
        "camera_label": "Take a photo of the leaf",
        "sample_btn": "🎲 Pick a random sample leaf",
        "sample_caption": "Loads a random image from the held-out test set.",
        "sample_missing": "No local test images found in `data/test/`.",
        "sample_loaded": "Sample loaded from the test set",

        # Image quality
        "quality_title": "Image quality check",
        "quality_leafcov": "Leaf coverage",
        "quality_good": "Frame looks good — the leaf fills enough of the image.",
        "quality_low": "The leaf fills only a small part of the frame. Move closer for a better reading.",
        "quality_blur": "Sharpness",
        "quality_blurry": "Image looks soft/blurry — accuracy may drop.",

        # Lesion analysis
        "lesion_title": "🔬 Lesion Area Analysis",
        "lesion_caption": "Measured directly from the pixels: share of the leaf surface that is discoloured, necrotic or spotted.",
        "lesion_affected": "Affected leaf area",
        "lesion_overlay": "Detected lesions (red)",
        "lesion_healthy_tissue": "Healthy tissue",
        "lesion_none": "No significant discolouration detected.",
        "lesion_sev_low": "Mild",
        "lesion_sev_mid": "Moderate",
        "lesion_sev_high": "Extensive",

        # Reliability / OOD
        "reliab_title": "🛡️ Prediction Reliability",
        "reliab_entropy": "Uncertainty (normalised entropy)",
        "reliab_margin": "Top-1 / Top-2 margin",
        "reliab_ok": "The model is confident and the decision is well separated.",
        "reliab_ood": "This image may be outside the 38 supported classes (or not a single leaf). Treat the result as a guess.",
        "reliab_close": "The top two candidates are very close — check both before acting.",

        # Report
        "report_title": "📄 Export",
        "report_btn": "⬇️ Download full report (HTML)",
        "report_caption": "A self-contained report with the image, scores, and care advice — opens in any browser or prints to PDF.",
        "report_heading": "Plant Health Report",
        "report_generated": "Generated",

        # History
        "hist_save": "💾 Save this scan to history",
        "hist_saved": "Scan saved to history.",
        "hist_title": "🕒 Scan History",
        "hist_caption": "Every saved scan is stored locally in a SQLite database (`data/plantvision_history.db`).",
        "hist_empty": "No scans saved yet. Analyse a leaf and press “Save this scan to history”.",
        "hist_total": "Saved scans",
        "hist_healthy": "Healthy",
        "hist_diseased": "Diseased",
        "hist_avg": "Average health score",
        "hist_trend": "Health score over time",
        "hist_dist": "Most scanned crops",
        "hist_clear": "🗑️ Clear all history",
        "hist_cleared": "History cleared.",
        "hist_csv": "⬇️ Export history (CSV)",

        # Batch
        "batch_title": "📦 Batch Scan",
        "batch_caption": "Analyse many leaves at once — useful for a whole field sample or a lab set.",
        "batch_upload": "Upload multiple leaf images",
        "batch_run": "🚀 Analyse all images",
        "batch_none": "Upload two or more images to run a batch scan.",
        "batch_progress": "Analysing image",
        "batch_summary": "Batch summary",
        "batch_healthy_ratio": "Healthy share",
        "batch_avg_conf": "Average confidence",
        "batch_avg_score": "Average health score",
        "batch_table": "Per-image results",
        "batch_csv": "⬇️ Download results (CSV)",
        "batch_chart_status": "Healthy vs diseased",
        "batch_chart_crop": "Detected crops",

        # Knowledge base
        "kb_title": "📚 Agricultural Knowledge Base",
        "kb_caption": "Searchable notes for all 38 supported conditions — symptoms, severity and treatment.",
        "kb_search": "Search symptoms, disease or crop",
        "kb_placeholder": "e.g. yellow spots on tomato leaves",
        "kb_filter": "Filter by crop",
        "kb_all": "All crops",
        "kb_no_results": "Nothing matched that search. Try fewer or simpler words.",
        "kb_results": "matches",
        "kb_severity": "Severity",
        "kb_advice": "Treatment & care",
        "kb_relevance": "Relevance",

        # Misc
        "sidebar_preview_note": "**Live now:** classifier · Grad-CAM · lesion-area analysis · health score · scan history · batch scan · knowledge base.\n\n**Planned:** pest detection (YOLO) and the RAG recommendation engine.",
        "top5_chart": "Probability distribution",
        "confidence_title": "Confidence",
        "reset_btn": "🔄 New scan",
    },
    "ar": {
        "tab_batch": "📦 فحص جماعي",
        "tab_history": "🕒 السجل",
        "tab_kb": "📚 قاعدة المعرفة",

        "src_title": "مصدر الصورة",
        "src_upload": "📁 رفع صورة",
        "src_camera": "📷 الكاميرا",
        "src_sample": "🎲 عينة جاهزة",
        "camera_label": "صوّر الورقة بالكاميرا",
        "sample_btn": "🎲 اختر عينة عشوائية",
        "sample_caption": "يحمّل صورة عشوائية من مجموعة الاختبار.",
        "sample_missing": "لا توجد صور اختبار محلية في `data/test/`.",
        "sample_loaded": "تم تحميل عينة من مجموعة الاختبار",

        "quality_title": "فحص جودة الصورة",
        "quality_leafcov": "نسبة الورقة في الصورة",
        "quality_good": "الإطار جيد — الورقة تملأ مساحة كافية من الصورة.",
        "quality_low": "الورقة تشغل جزءًا صغيرًا من الصورة. اقترب أكثر للحصول على نتيجة أدق.",
        "quality_blur": "درجة الوضوح",
        "quality_blurry": "الصورة تبدو غير واضحة — قد تقل دقة النتيجة.",

        "lesion_title": "🔬 تحليل مساحة الإصابة",
        "lesion_caption": "قياس مباشر من البكسلات: نسبة سطح الورقة المصاب بالتغير اللوني أو التنقيط أو التنخر.",
        "lesion_affected": "نسبة المساحة المصابة",
        "lesion_overlay": "المناطق المصابة (بالأحمر)",
        "lesion_healthy_tissue": "نسيج سليم",
        "lesion_none": "لم يتم رصد تغير لوني ملحوظ.",
        "lesion_sev_low": "خفيفة",
        "lesion_sev_mid": "متوسطة",
        "lesion_sev_high": "واسعة",

        "reliab_title": "🛡️ موثوقية التنبؤ",
        "reliab_entropy": "درجة عدم اليقين (إنتروبيا مُطبّعة)",
        "reliab_margin": "الفارق بين أعلى احتمالين",
        "reliab_ok": "النموذج واثق والقرار واضح ومنفصل عن باقي الاحتمالات.",
        "reliab_ood": "قد تكون الصورة خارج الـ 38 فئة المدعومة (أو ليست ورقة مفردة). تعامل مع النتيجة كتخمين.",
        "reliab_close": "أعلى احتمالين متقاربان جدًا — راجع الاثنين قبل اتخاذ أي إجراء.",

        "report_title": "📄 تصدير",
        "report_btn": "⬇️ تحميل التقرير الكامل (HTML)",
        "report_caption": "تقرير مستقل يحتوي الصورة والدرجات ونصائح العلاج — يفتح في أي متصفح ويمكن طباعته PDF.",
        "report_heading": "تقرير صحة النبات",
        "report_generated": "تاريخ الإصدار",

        "hist_save": "💾 احفظ هذا الفحص في السجل",
        "hist_saved": "تم حفظ الفحص في السجل.",
        "hist_title": "🕒 سجل الفحوصات",
        "hist_caption": "كل فحص محفوظ يُخزَّن محليًا في قاعدة بيانات SQLite (‏`data/plantvision_history.db`).",
        "hist_empty": "لا توجد فحوصات محفوظة بعد. حلّل ورقة ثم اضغط «احفظ هذا الفحص في السجل».",
        "hist_total": "عدد الفحوصات",
        "hist_healthy": "سليمة",
        "hist_diseased": "مصابة",
        "hist_avg": "متوسط درجة الصحة",
        "hist_trend": "تطور درجة الصحة بمرور الوقت",
        "hist_dist": "أكثر المحاصيل فحصًا",
        "hist_clear": "🗑️ مسح السجل بالكامل",
        "hist_cleared": "تم مسح السجل.",
        "hist_csv": "⬇️ تصدير السجل (CSV)",

        "batch_title": "📦 الفحص الجماعي",
        "batch_caption": "حلّل عدة أوراق مرة واحدة — مفيد لعينة حقل كاملة أو مجموعة معملية.",
        "batch_upload": "ارفع عدة صور لأوراق",
        "batch_run": "🚀 حلّل كل الصور",
        "batch_none": "ارفع صورتين أو أكثر لبدء الفحص الجماعي.",
        "batch_progress": "جاري تحليل الصورة",
        "batch_summary": "ملخص الفحص",
        "batch_healthy_ratio": "نسبة السليم",
        "batch_avg_conf": "متوسط الثقة",
        "batch_avg_score": "متوسط درجة الصحة",
        "batch_table": "نتائج كل صورة",
        "batch_csv": "⬇️ تحميل النتائج (CSV)",
        "batch_chart_status": "سليم مقابل مصاب",
        "batch_chart_crop": "المحاصيل المكتشفة",

        "kb_title": "📚 قاعدة المعرفة الزراعية",
        "kb_caption": "ملاحظات قابلة للبحث لكل الحالات الـ 38 — الأعراض والشدة والعلاج.",
        "kb_search": "ابحث بالأعراض أو المرض أو المحصول",
        "kb_placeholder": "مثال: بقع صفراء على أوراق الطماطم",
        "kb_filter": "تصفية حسب المحصول",
        "kb_all": "كل المحاصيل",
        "kb_no_results": "لا توجد نتائج مطابقة. جرّب كلمات أقل أو أبسط.",
        "kb_results": "نتيجة",
        "kb_severity": "الشدة",
        "kb_advice": "العلاج والرعاية",
        "kb_relevance": "درجة التطابق",

        "sidebar_preview_note": "**متاح الآن:** المصنّف · Grad-CAM · تحليل مساحة الإصابة · درجة الصحة · سجل الفحوصات · الفحص الجماعي · قاعدة المعرفة.\n\n**قيد التطوير:** كشف الآفات (YOLO) ومحرك التوصيات RAG.",
        "top5_chart": "توزيع الاحتمالات",
        "confidence_title": "نسبة الثقة",
        "reset_btn": "🔄 فحص جديد",
    },
}

for _lang, _strings in EXTRA_TRANSLATIONS.items():
    TRANSLATIONS.setdefault(_lang, {}).update(_strings)


def tr(key: str, lang: str = "en") -> str:
    """Get a translated string from TRANSLATIONS."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ============================================================================
# Health Score
# ============================================================================
def calculate_health_score(top_class: str, confidence: float) -> int:
    """
    Compute a 0-100 health score.

    - Healthy classes → score based on confidence (85-100)
    - Moderate severity diseases → score 40-65
    - Severe diseases → score 5-40

    All values are deterministic from the classifier output; no fabricated numbers.
    """
    _, _, is_healthy = parse_class(top_class)
    info = DISEASE_INFO.get(top_class, {})
    severity = info.get("severity", "Moderate")

    if is_healthy:
        # Healthy leaves: 85 to 100 based on confidence
        return int(85 + 15 * confidence)

    if severity == "Severe":
        # Severe: 5-40 range
        return int(5 + 35 * (1 - confidence))

    # Moderate: 40-65 range
    return int(40 + 25 * (1 - confidence))


def health_score_gauge(score: int, lang: str = "en") -> str:
    """Return HTML for the health score gauge."""
    # Color based on score
    if score >= 80:
        color = "#4ade80"
        label_key = "hs_healthy"
        desc_key = "hs_desc_healthy"
    elif score >= 50:
        color = "#fbbf24"
        label_key = "hs_moderate"
        desc_key = "hs_desc_moderate"
    else:
        color = "#f87171"
        label_key = "hs_critical"
        desc_key = "hs_desc_critical"

    label = tr(label_key, lang)
    desc = tr(desc_key, lang)

    # SVG circular progress
    circumference = 2 * 3.14159 * 45  # ~282.74
    offset = circumference * (1 - score / 100)

    return f"""
    <div style="text-align:center;padding:1rem 0;">
      <svg width="160" height="160" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="45" stroke="rgba(74,222,128,0.15)"
                stroke-width="10" fill="none"/>
        <circle cx="60" cy="60" r="45" stroke="{color}"
                stroke-width="10" fill="none" stroke-linecap="round"
                stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}"
                transform="rotate(-90 60 60)"
                style="transition: stroke-dashoffset 0.8s ease;"/>
        <text x="60" y="58" text-anchor="middle" fill="#f0fdf4"
              font-size="30" font-weight="700">{score}</text>
        <text x="60" y="76" text-anchor="middle" fill="{color}"
              font-size="11" font-weight="700" letter-spacing="1">
            {label.upper()}
        </text>
      </svg>
      <p style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem;">{desc}</p>
    </div>
    """


# ============================================================================
# NEW · Lesion area analysis  (measured, not guessed)
# ============================================================================
def analyze_lesions(pil_image: Image.Image) -> dict:
    """
    Measure how much of the leaf surface is discoloured / necrotic.

    Method (pure OpenCV, fully deterministic):
      1. Segment leaf pixels in HSV (green + brown/yellow tissue).
      2. Inside that leaf mask, mark pixels that are NOT healthy green
         (brown, yellow, grey, black necrosis) as lesion pixels.
      3. affected % = lesion pixels / leaf pixels.

    Returns dict with:
      affected_pct   float  0-100, share of leaf tissue that looks damaged
      leaf_coverage  float  0-100, share of the frame occupied by the leaf
      overlay        np.ndarray RGB image with lesions tinted red
      leaf_px        int    number of leaf pixels found
    """
    img = np.array(pil_image.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # --- leaf tissue: anything green / yellow / brown with enough saturation ---
    leaf = ((s_ch > 35) & (v_ch > 25) & (h_ch >= 5) & (h_ch <= 95)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, kernel)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, kernel)

    leaf_px = int(leaf.sum())
    total_px = img.shape[0] * img.shape[1]
    if leaf_px < 0.01 * total_px:
        return {
            "affected_pct": 0.0,
            "leaf_coverage": 100.0 * leaf_px / max(total_px, 1),
            "overlay": img,
            "leaf_px": leaf_px,
        }

    # --- healthy green tissue inside the leaf ---
    healthy = ((h_ch >= 33) & (h_ch <= 92) & (s_ch > 55) & (v_ch > 45)).astype(np.uint8)

    # --- necrotic / very dark tissue inside the leaf ---
    necrotic = ((v_ch < 55) | ((s_ch < 45) & (v_ch < 190))).astype(np.uint8)

    lesion = (leaf.astype(bool) & (~healthy.astype(bool) | necrotic.astype(bool))).astype(np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    affected_pct = 100.0 * float(lesion.sum()) / float(leaf_px)

    # --- red overlay for the UI ---
    overlay = img.copy()
    red = np.zeros_like(img)
    red[:, :, 0] = 255
    mask3 = lesion.astype(bool)
    overlay[mask3] = (0.55 * red[mask3] + 0.45 * overlay[mask3]).astype(np.uint8)

    # thin green outline around the detected leaf, so the user sees the segmentation
    contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (74, 222, 128), 2)

    return {
        "affected_pct": round(min(affected_pct, 100.0), 1),
        "leaf_coverage": round(100.0 * leaf_px / total_px, 1),
        "overlay": overlay,
        "leaf_px": leaf_px,
    }


def lesion_band(pct: float, lang: str = "en") -> tuple[str, str]:
    """Map an affected-area percentage to a (label, colour) pair."""
    if pct < 8:
        return tr("lesion_sev_low", lang), "#4ade80"
    if pct < 30:
        return tr("lesion_sev_mid", lang), "#fbbf24"
    return tr("lesion_sev_high", lang), "#f87171"


def refine_health_score(base_score: int, affected_pct: float, is_healthy: bool) -> int:
    """
    Blend the classifier-based score with the measured lesion area.

    The classifier score stays dominant (70%); the measured damaged area
    contributes the remaining 30%. Both inputs are real — nothing invented.
    """
    area_score = max(0.0, 100.0 - 2.2 * affected_pct)
    blended = 0.7 * base_score + 0.3 * area_score
    if is_healthy:
        blended = max(blended, base_score - 12)  # don't over-punish a healthy call
    return int(max(0, min(100, round(blended))))


# ============================================================================
# NEW · Image quality check
# ============================================================================
def image_quality(pil_image: Image.Image, leaf_coverage: float) -> dict:
    """Blur (variance of Laplacian) + leaf framing check."""
    gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "sharpness": round(sharpness, 1),
        "is_blurry": sharpness < 80.0,
        "leaf_coverage": leaf_coverage,
        "poor_framing": leaf_coverage < 20.0,
    }


# ============================================================================
# NEW · Prediction reliability (out-of-distribution guard)
# ============================================================================
def reliability_stats(probs: np.ndarray) -> dict:
    """
    Turn the full softmax vector into interpretable reliability numbers.

    normalised entropy ≈ 0  → one class dominates (confident)
    normalised entropy ≈ 1  → the model is spreading probability everywhere,
                              which usually means the image is not one of the
                              38 supported leaf classes.
    """
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    entropy = float(-(p * np.log(p)).sum())
    norm_entropy = entropy / math.log(len(p))
    order = np.sort(p)[::-1]
    margin = float(order[0] - order[1])
    return {
        "entropy": round(norm_entropy, 3),
        "margin": round(margin, 3),
        "is_ood": norm_entropy > 0.45 or order[0] < 0.45,
        "is_close": margin < 0.15,
    }

def predict_detailed(model, idx_to_class, device, image: Image.Image, k: int = 5):
    """Like predict(), but also returns the full probability vector."""
    normalized = normalize_to_plantvillage(image)  # ← أضيف هذا السطر
    tf = build_transform()
    x = tf(normalized).unsqueeze(0).to(device)  # ← غيّر image إلى normalized
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()
    top_probs, top_idx = torch.topk(probs, k=min(k, len(idx_to_class)))
    top = [(idx_to_class[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]
    return top, probs.numpy()




def enhanced_reliability(
    probs: np.ndarray,
    leaf_coverage: float,
    sharpness: float,
    affected_pct: float,
) -> dict:
    """Reliability including image quality metrics."""
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    entropy = float(-(p * np.log(p)).sum())
    norm_entropy = entropy / math.log(len(p))
    order = np.sort(p)[::-1]
    margin = float(order[0] - order[1])
    
    return {
        "entropy": round(norm_entropy, 3),
        "margin": round(margin, 3),
        "is_ood": norm_entropy > 0.40 or order[0] < 0.45,
        "is_close": margin < 0.12,
        "leaf_coverage": round(leaf_coverage, 1),
        "sharpness": round(sharpness, 1),
        "affected_pct": round(affected_pct, 1),
        "is_clear": sharpness >= 110.0 and leaf_coverage >= 20.0,
    }
# ============================================================================
# NEW · Plotly visuals
# ============================================================================
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
)


def plotly_confidence_gauge(prob: float, is_healthy: bool, title: str):
    """Animated-looking Plotly gauge for the top-1 confidence."""
    color = "#4ade80" if is_healthy else "#fbbf24" if prob < 0.9 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "#f0fdf4"}},
        title={"text": title, "font": {"size": 13, "color": "#86efac"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(248,113,113,0.12)"},
                {"range": [50, 80], "color": "rgba(251,191,36,0.12)"},
                {"range": [80, 100], "color": "rgba(74,222,128,0.14)"},
            ],
            "threshold": {
                "line": {"color": "#f0fdf4", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10), **PLOTLY_DARK)
    return fig


def plotly_top5_bars(results, lang: str, ar_names: dict):
    """Horizontal bar chart of the top-k candidate probabilities."""
    rows = []
    for raw, prob in results:
        p, c, healthy = parse_class(raw)
        if lang == "ar" and raw in ar_names:
            p, c = ar_names[raw]
        rows.append({
            "label": f"{p} — {tr('healthy', lang) if healthy else c}",
            "prob": prob * 100,
        })
    df = pd.DataFrame(rows).iloc[::-1]
    fig = px.bar(
        df, x="prob", y="label", orientation="h",
        color="prob", color_continuous_scale=["#1a4d2e", "#22c55e", "#86efac"],
        range_color=(0, 100), text=df["prob"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=300, showlegend=False, coloraxis_showscale=False,
        xaxis_title="", yaxis_title="", xaxis_range=[0, 115],
        margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_DARK,
    )
    return fig


def plotly_area_donut(affected_pct: float, lang: str):
    """Donut chart: healthy tissue vs measured lesion area."""
    fig = go.Figure(go.Pie(
        labels=[tr("lesion_healthy_tissue", lang), tr("lesion_affected", lang)],
        values=[max(0.0, 100 - affected_pct), affected_pct],
        hole=0.62,
        marker=dict(colors=["#22c55e", "#f87171"], line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent",
    ))
    fig.update_layout(
        height=260, showlegend=True,
        legend=dict(orientation="h", y=-0.1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text=f"<b>{affected_pct:.1f}%</b>", x=0.5, y=0.5,
            font=dict(size=24, color="#f0fdf4"), showarrow=False,
        )],
        **PLOTLY_DARK,
    )
    return fig


# ============================================================================
# NEW · Scan history (local SQLite)
# ============================================================================
@st.cache_resource
def get_history_conn():
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(HISTORY_DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,
            plant         TEXT    NOT NULL,
            condition     TEXT    NOT NULL,
            is_healthy    INTEGER NOT NULL,
            confidence    REAL    NOT NULL,
            health_score  INTEGER NOT NULL,
            affected_pct  REAL    NOT NULL,
            raw_class     TEXT    NOT NULL,
            thumb_b64     TEXT
        )
    """)
    conn.commit()
    return conn


def thumb_to_b64(pil_image: Image.Image, size: int = 112) -> str:
    """Small JPEG thumbnail as a base64 string (for history + reports)."""
    im = pil_image.convert("RGB").copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_b64(pil_image: Image.Image, max_side: int = 640) -> str:
    """Larger JPEG (used inside the downloadable report)."""
    im = pil_image.convert("RGB").copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_scan(plant, condition, is_healthy, confidence, health_score,
              affected_pct, raw_class, pil_image):
    conn = get_history_conn()
    conn.execute(
        "INSERT INTO scans (ts, plant, condition, is_healthy, confidence, "
        "health_score, affected_pct, raw_class, thumb_b64) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plant, condition, int(is_healthy), float(confidence),
            int(health_score), float(affected_pct), raw_class,
            thumb_to_b64(pil_image),
        ),
    )
    conn.commit()


def load_history() -> pd.DataFrame:
    conn = get_history_conn()
    return pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC", conn)


def clear_history():
    conn = get_history_conn()
    conn.execute("DELETE FROM scans")
    conn.commit()


# ============================================================================
# NEW · Downloadable health report
# ============================================================================
def build_report_html(*, lang, plant, condition, is_healthy, confidence,
                      health_score, affected_pct, info, results, image_b64,
                      overlay_b64, reliability) -> str:
    """Self-contained HTML report — no external assets, prints straight to PDF."""
    direction = "rtl" if lang == "ar" else "ltr"
    status_color = "#16a34a" if is_healthy else "#dc2626"
    status_text = tr("healthy", lang) if is_healthy else condition
    rows = ""
    for i, (raw, prob) in enumerate(results, start=1):
        p, c, healthy = parse_class(raw)
        if lang == "ar" and raw in AR_CLASS_NAMES:
            p, c = AR_CLASS_NAMES[raw]
        label = tr("healthy", lang) if healthy else c
        rows += (f"<tr><td>{i}</td><td>{p}</td><td>{label}</td>"
                 f"<td style='text-align:end'>{prob:.2%}</td></tr>")

    desc = info.get("description", "") if info else ""
    advice = info.get("advice", "") if info else ""
    severity = info.get("severity", "—") if info else "—"

    return f"""<!doctype html>
<html dir="{direction}" lang="{lang}">
<head>
<meta charset="utf-8">
<title>PlantVision AI — {tr('report_heading', lang)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Tahoma, sans-serif;
         background:#f8fafc; color:#0f172a; margin:0; padding:32px; }}
  .wrap {{ max-width: 860px; margin:0 auto; background:#fff; border-radius:18px;
          box-shadow:0 10px 40px rgba(15,23,42,.08); overflow:hidden; }}
  header {{ background:linear-gradient(135deg,#166534,#22c55e); color:#f0fdf4; padding:28px 32px; }}
  header h1 {{ margin:0; font-size:1.7rem; }}
  header p  {{ margin:6px 0 0; opacity:.9; font-size:.9rem; }}
  .body {{ padding:28px 32px; }}
  .grid {{ display:flex; gap:24px; flex-wrap:wrap; }}
  .grid img {{ width:300px; border-radius:14px; border:1px solid #e2e8f0; }}
  .status {{ font-size:1.5rem; font-weight:800; color:{status_color}; margin:0 0 4px; }}
  .kv {{ margin:14px 0; }}
  .kv div {{ display:flex; justify-content:space-between; padding:8px 0;
             border-bottom:1px dashed #e2e8f0; font-size:.94rem; }}
  .kv b {{ color:#166534; }}
  h3 {{ margin:26px 0 8px; color:#166534; font-size:1.05rem; }}
  p.note {{ font-size:.92rem; line-height:1.6; color:#334155; margin:0; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:.9rem; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:start; }}
  th {{ background:#f1f5f9; color:#166534; }}
  footer {{ padding:18px 32px; background:#f8fafc; color:#64748b; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
  @media print {{ body {{ background:#fff; padding:0; }} .wrap {{ box-shadow:none; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🌿 PlantVision AI — {tr('report_heading', lang)}</h1>
    <p>{tr('report_generated', lang)}: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  </header>
  <div class="body">
    <div class="grid">
      <div>
        <img src="data:image/jpeg;base64,{image_b64}" alt="leaf">
        {'<img src="data:image/jpeg;base64,' + overlay_b64 + '" alt="lesions" style="margin-top:12px">' if overlay_b64 else ''}
      </div>
      <div style="flex:1; min-width:260px;">
        <p class="status">{plant}</p>
        <p style="margin:0;color:#475569;">{status_text}</p>
        <div class="kv">
          <div><span>{tr('confidence_title', lang)}</span><b>{confidence:.2%}</b></div>
          <div><span>{tr('health_score', lang)}</span><b>{health_score}/100</b></div>
          <div><span>{tr('lesion_affected', lang)}</span><b>{affected_pct:.1f}%</b></div>
          <div><span>{tr('severity', lang)}</span><b>{severity}</b></div>
          <div><span>{tr('reliab_entropy', lang)}</span><b>{reliability['entropy']:.3f}</b></div>
        </div>
      </div>
    </div>

    <h3>{tr('condition', lang)}</h3>
    <p class="note">{desc}</p>

    <h3>{tr('kb_advice', lang)}</h3>
    <p class="note">{advice}</p>

    <h3>{tr('top5_title', lang)}</h3>
    <table>
      <tr><th>#</th><th>{tr('sidebar_classes', lang)}</th><th>{tr('condition', lang)}</th><th style="text-align:end">%</th></tr>
      {rows}
    </table>
  </div>
  <footer>{tr('disclaimer', lang)}</footer>
</div>
</body>
</html>"""


# ============================================================================
# NEW · Knowledge base search (TF-IDF over the disease notes)
# ============================================================================
@st.cache_resource
def build_kb_index():
    """Vectorise every disease note once so the Knowledge tab can search them."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    keys, docs = [], []
    for raw, info in DISEASE_INFO.items():
        plant, condition, _ = parse_class(raw)
        docs.append(f"{plant} {condition} {info.get('severity','')} "
                    f"{info.get('description','')} {info.get('advice','')}")
        keys.append(raw)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(docs)
    return vec, matrix, keys


def search_kb(query: str, top_k: int = 6):
    """Return [(raw_class, score)] ranked by relevance to the query."""
    from sklearn.metrics.pairwise import cosine_similarity

    vec, matrix, keys = build_kb_index()
    q = vec.transform([query])
    scores = cosine_similarity(q, matrix)[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [(keys[i], float(scores[i])) for i in order if scores[i] > 0.01]


# ============================================================================
# NEW · Sample leaves from the held-out test set
# ============================================================================
@st.cache_data
def list_sample_images(limit_per_class: int = 2):
    """Collect a few test images so the app is demoable without user photos."""
    if not SAMPLES_DIR.exists():
        return []
    paths = []
    for cls_dir in sorted(SAMPLES_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in sorted(cls_dir.glob("*"))
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        paths.extend(imgs[:limit_per_class])
    return [str(p) for p in paths]


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
# GUARD RAILS (NEW) · Confidence guards for field images
# ============================================================================

def normalize_to_plantvillage(pil_image: Image.Image) -> Image.Image:
    """Histogram normalization for field photos."""
    img = np.array(pil_image.convert("RGB"))
    if img.shape[0] < 64 or img.shape[1] < 64:
        return pil_image
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab_adjusted = cv2.merge([l, a, b])
    rgb_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(rgb_adjusted.astype(np.uint8))


def get_class_confidence_threshold(raw_class: str, metrics: dict) -> float:
    """Per-class confidence threshold based on F1 scores."""
    if not metrics or "per_class_f1" not in metrics:
        return 0.85
    
    f1 = metrics["per_class_f1"].get(raw_class, 0.85)
    if f1 > 0.95:
        return 0.70
    elif f1 > 0.85:
        return 0.78
    else:
        return 0.86


def should_trust_prediction(
    top_prob: float,
    entropy: float,
    margin: float,
    leaf_coverage: float,
    sharpness: float,
    is_healthy: bool,
    class_threshold: float,
) -> tuple[bool, str]:
    """
    Decide whether a prediction is safe to present as a real answer.

    Two levels:
      BLOCKERS  — the model itself is not sure enough. Any one of these
                  fails the gate:
                    * confidence below the per-class threshold
                    * normalised entropy above 0.40 (probability spread out)
                    * top-1/top-2 margin below 0.12 (no clear winner)
      WARNINGS  — the photo is poor (blurry, leaf too small). One warning on
                  its own does not block, because PlantVillage-style photos are
                  often soft; two warnings together do.

    Returns (should_trust, reason). The reason lists every issue found.
    """
    blockers, warnings = [], []

    if top_prob < class_threshold:
        blockers.append(
            f"Confidence {top_prob:.1%} is below the {class_threshold:.0%} bar for this class"
        )
    if entropy > 0.40:
        blockers.append(
            f"The model spread its probability across many classes (uncertainty {entropy:.2f})"
        )
    if margin < 0.12:
        blockers.append(
            f"Top-1 and top-2 are almost tied (gap {margin:.2f}) — no clear winner"
        )

    if leaf_coverage < 15.0:
        warnings.append(f"the leaf fills only {leaf_coverage:.0f}% of the frame")
    if sharpness < 60.0:
        warnings.append(f"the image is soft/blurry (sharpness {sharpness:.0f})")

    failed = list(blockers)
    if len(warnings) >= 2:
        failed.append("photo quality is poor: " + " and ".join(warnings))
    elif warnings and blockers:
        failed.append("also, " + warnings[0])

    if failed:
        return False, " · ".join(failed) + "."
    return True, ""


def consensus_prediction(
    original: Image.Image,
    model,
    idx_to_class,
    device,
    auto_crop_enabled: bool = True,
) -> dict:
    """Predict on multiple crops, compare votes."""
    w, h = original.size
    
    if auto_crop_enabled:
        center_crop = auto_crop_leaf(original)
    else:
        center_crop = original
    
    crops = [center_crop]
    
    crop_size_w = int(w * 0.65)
    crop_size_h = int(h * 0.65)
    
    corners = [
        (0, 0),
        (w - crop_size_w, 0),
        (0, h - crop_size_h),
        (w - crop_size_w, h - crop_size_h),
    ]
    
    for x, y in corners:
        if x >= 0 and y >= 0 and x + crop_size_w <= w and y + crop_size_h <= h:
            corner_crop = original.crop((x, y, x + crop_size_w, y + crop_size_h))
            crops.append(corner_crop)
    
    all_results = []
    vote_dict = {}
    
    for crop in crops:
        res, _ = predict_detailed(model, idx_to_class, device, crop, k=1)
        if res:
            top_class, prob = res[0]
            all_results.append((top_class, prob))
            
            plant, _, _ = parse_class(top_class)
            vote_dict[plant] = vote_dict.get(plant, 0) + 1
    
    best_plant = max(vote_dict.items(), key=lambda x: x[1])[0]
    votes = vote_dict[best_plant]
    
    center_results, _ = predict_detailed(model, idx_to_class, device, center_crop, k=5)
    
    consensus_class = center_results[0][0] if center_results else ""
    top_prob = center_results[0][1] if center_results else 0.0
    
    return {
        "consensus_class": consensus_class,
        "consensus_plant": best_plant,
        "votes": votes,
        "total_crops": len(crops),
        "top_prob": top_prob,
        "full_results": center_results,
        "all_results": all_results,
    }


# ============================================================================
# Language selector (defined BEFORE header so it's available everywhere)
# ============================================================================
lang_choice = st.radio(
    "🌐 Language / اللغة",
    options=["en", "ar"],
    format_func=lambda x: "English" if x == "en" else "العربية",
    horizontal=True,
    key="lang_choice",
)
lang = lang_choice

if lang == "ar":
    st.markdown(RTL_CSS, unsafe_allow_html=True)


# ============================================================================
# Load model and metrics
# ============================================================================
model, idx_to_class, device, ckpt = load_model()
metrics = load_metrics()


# ============================================================================
# Header — Hero section
# ============================================================================
_acc_chip = (f"<span class='pv-chip'>🎯 {metrics['accuracy']:.1%} {tr('sidebar_acc', lang)}</span>"
             if metrics else "")
_top5_chip = (f"<span class='pv-chip'>🏅 {metrics['top5_accuracy']:.1%} Top-5</span>"
              if metrics else "")

LEAF_MARK = (
    '<svg width="88" height="88" viewBox="0 0 64 64" fill="none" style="flex-shrink:0">'
    '<defs><linearGradient id="pvLeaf" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0%" stop-color="#ecfdf5"/><stop offset="55%" stop-color="#86efac"/>'
    '<stop offset="100%" stop-color="#16a34a"/></linearGradient>'
    '<filter id="pvGlow" x="-50%" y="-50%" width="200%" height="200%">'
    '<feGaussianBlur stdDeviation="2.5" result="b"/>'
    '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
    '<circle cx="32" cy="32" r="30" fill="rgba(255,255,255,0.07)" '
    'stroke="rgba(187,247,208,0.4)" stroke-width="1.5"/>'
    '<circle cx="32" cy="32" r="24" fill="none" stroke="rgba(187,247,208,0.18)" '
    'stroke-width="1" stroke-dasharray="3 5"/>'
    '<path d="M46 15C29 15 17 24 17 38c0 4.2 1.5 7.8 3.9 10.6C25.6 38 33 30.6 44 26.2'
    'c-7.6 5.2-13 11.6-16.2 23.4 15 2 23.2-8.4 23.2-23 0-4.2-1.3-8.4-5-11.6z" '
    'fill="url(#pvLeaf)" filter="url(#pvGlow)"/>'
    '<path d="M20.9 48.6C25.6 38 33 30.6 44 26.2" stroke="#0b2016" stroke-width="1.7" '
    'stroke-linecap="round" opacity="0.5"/>'
    '</svg>'
)

_hero_html = (
    '<div class="pv-hero">'
    '<div class="pv-hero-top">'
    + LEAF_MARK +
    '<div class="pv-hero-copy">'
    '<div class="pv-hero-eyebrow">' + tr("sidebar_subtitle", lang) + '</div>'
    '<h1>PlantVision&nbsp;AI</h1>'
    '<p>' + tr("hero_tagline", lang) + '</p>'
    '<p class="pv-hero-sub">' + tr("hero_subtitle", lang) + '</p>'
    '</div></div>'
    '<div class="pv-hero-grid">'
    '<span class="pv-badge">' + tr("hero_badge", lang) + '</span>'
    + _acc_chip + _top5_chip +
    '<span class="pv-chip">\U0001F33E ' + f"{len(idx_to_class)}" + ' ' + tr("sidebar_classes", lang) + '</span>'
    '<span class="pv-chip pv-chip-info">\u2699\uFE0F ' + str(device) + '</span>'
    '</div></div>'
)

st.markdown(_hero_html, unsafe_allow_html=True)


# ============================================================================
# Sidebar — Model info
# ============================================================================
with st.sidebar:
    st.markdown(f"### 🌿 {tr('sidebar_title', lang)}")
    st.caption(tr("sidebar_subtitle", lang))
    st.divider()

    st.markdown(f"#### 🧠 {tr('sidebar_model', lang)}")
    st.markdown(f"""
    <div class="pv-metric-card">
        <div class="pv-metric-label">{tr('sidebar_arch', lang)}</div>
        <div class="pv-metric-value" style="font-size:1.1rem;">MobileNetV2</div>
    </div>
    <div class="pv-metric-card">
        <div class="pv-metric-label">{tr('sidebar_device', lang)}</div>
        <div class="pv-metric-value" style="font-size:1.1rem;">{device}</div>
    </div>
    <div class="pv-metric-card">
        <div class="pv-metric-label">{tr('sidebar_classes', lang)}</div>
        <div class="pv-metric-value">{len(idx_to_class)}</div>
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        st.markdown(f"#### 📊 {tr('sidebar_perf', lang)}")
        st.markdown(f"""
        <div class="pv-metric-card">
            <div class="pv-metric-label">{tr('sidebar_acc', lang)}</div>
            <div class="pv-metric-value">{metrics['accuracy']:.2%}</div>
        </div>
        <div class="pv-metric-card">
            <div class="pv-metric-label">{tr('sidebar_top5', lang)}</div>
            <div class="pv-metric-value">{metrics['top5_accuracy']:.2%}</div>
        </div>
        <div class="pv-metric-card">
            <div class="pv-metric-label">{tr('sidebar_f1', lang)}</div>
            <div class="pv-metric-value">{metrics['f1_macro']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"{tr('sidebar_evaluated', lang)} {metrics['num_test_images']:,} {tr('sidebar_evaluated_imgs', lang)}")

    st.divider()
    st.caption(tr("sidebar_preview_note", lang))


# ============================================================================
# Tabs
# ============================================================================
(tab_predict, tab_batch, tab_history,
 tab_kb, tab_info, tab_analytics) = st.tabs([
    tr("tab_predict", lang),
    tr("tab_batch", lang),
    tr("tab_history", lang),
    tr("tab_kb", lang),
    tr("tab_info", lang),
    tr("tab_analytics", lang),
])


# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    # ------------------------------------------------------------------
    # Input source: upload · camera · random sample from the test set
    # ------------------------------------------------------------------
    src = st.radio(
        tr("src_title", lang),
        options=["upload", "camera", "sample"],
        format_func=lambda s: {
            "upload": tr("src_upload", lang),
            "camera": tr("src_camera", lang),
            "sample": tr("src_sample", lang),
        }[s],
        horizontal=True,
        key="input_source",
    )

    original = None
    col_in, col_opt = st.columns([2, 1])

    with col_in:
        if src == "upload":
            uploaded = st.file_uploader(
                tr("upload_label", lang),
                type=["jpg", "jpeg", "png"],
                help=tr("upload_help", lang),
            )
            if uploaded is not None:
                original = Image.open(uploaded).convert("RGB")

        elif src == "camera":
            shot = st.camera_input(tr("camera_label", lang))
            if shot is not None:
                original = Image.open(shot).convert("RGB")

        else:
            samples = list_sample_images()
            if not samples:
                st.info(tr("sample_missing", lang))
            else:
                if st.button(tr("sample_btn", lang)) or "sample_path" not in st.session_state:
                    st.session_state["sample_path"] = random.choice(samples)
                st.caption(tr("sample_caption", lang))
                sample_path = st.session_state.get("sample_path")
                if sample_path and Path(sample_path).exists():
                    original = Image.open(sample_path).convert("RGB")
                    st.markdown(
                        f"<span class='pv-chip pv-chip-info'>🎲 {tr('sample_loaded', lang)} · "
                        f"{Path(sample_path).parent.name}</span>",
                        unsafe_allow_html=True,
                    )

    with col_opt:
        auto_crop = st.toggle(
            tr("autocrop_label", lang),
            value=True,
            help=tr("autocrop_help", lang),
        )
        st.caption(tr("autocrop_caption", lang))
        show_lesions = st.toggle(tr("lesion_title", lang), value=True)
        show_gradcam = st.toggle(tr("gradcam_title", lang), value=True)

    # --- Tips expander ---
    with st.expander(tr("tips_title", lang), expanded=False):
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

    # ------------------------------------------------------------------
    # Result section
    # ------------------------------------------------------------------
    if original is None:
        st.info(tr("no_image", lang))
    else:
        if auto_crop:
            with st.spinner(tr("detecting", lang)):
                processed = auto_crop_leaf(original)
            crop_applied = processed.size != original.size
        else:
            processed = original
            crop_applied = False

        # Low-res warning
        w, h = processed.size
        if min(w, h) < 200:
            st.warning(tr("low_res", lang).format(w=w, h=h))

        # --- Run inference with guards ---
        use_consensus = st.toggle(
            "\U0001F5F3\uFE0F Consensus voting (compare 5 crops)",
            value=False,
            key="use_consensus",
            help="Runs the model on 5 crops of the image and checks whether they agree.",
        )

        with st.spinner(tr("analyzing", lang)):
            results, prob_vector = predict_detailed(model, idx_to_class, device, processed)
            consensus_votes = consensus_total = None
            if use_consensus:
                consensus = consensus_prediction(
                    processed, model, idx_to_class, device, auto_crop
                )
                results = consensus["full_results"]
                consensus_votes = consensus["votes"]
                consensus_total = consensus["total_crops"]

        top_class, top_prob = results[0]
        plant, condition, is_healthy = parse_class(top_class)
        # Get per-class threshold
        class_threshold = get_class_confidence_threshold(top_class, metrics)
        
        # === Measured lesion area (BEFORE enhanced_reliability) ===
        lesion = analyze_lesions(processed)
        affected_pct = lesion["affected_pct"] if not is_healthy else min(lesion["affected_pct"], 100.0)
        quality = image_quality(processed, lesion["leaf_coverage"])
        
        # Enhanced reliability (with quality metrics)
        reliability = enhanced_reliability(prob_vector, lesion["leaf_coverage"], quality["sharpness"], affected_pct)
        
        # GUARD: Should we trust this?
        should_show, guard_reason = should_trust_prediction(
            top_prob=top_prob,
            entropy=reliability["entropy"],
            margin=reliability["margin"],
            leaf_coverage=reliability["leaf_coverage"],
            sharpness=reliability["sharpness"],
            is_healthy=is_healthy,
            class_threshold=class_threshold,
        )
        # Arabic name override
        if lang == "ar" and top_class in AR_CLASS_NAMES:
            plant, condition = AR_CLASS_NAMES[top_class]

        # --- Health score (classifier + measured damage) ---
        base_score = calculate_health_score(top_class, top_prob)
        health_score = refine_health_score(base_score, affected_pct, is_healthy)

        # === RELIABILITY GATE =================================================
        if should_show:
            st.success(f"✅ Prediction looks reliable — confidence {top_prob:.1%}")
            if consensus_votes:
                st.caption(
                    f"🗳️ Consensus: {consensus_votes}/{consensus_total} crops agree on this plant"
                )
            result_area = st.container()
        else:
            st.error(f"⚠️ **This prediction is not reliable.**\n\n{guard_reason}")
            st.info(
                "💡 The model was trained on clean single-leaf photos. For a trustworthy reading:\n"
                "- **One leaf only** — fill the frame\n"
                "- **Plain background** (white paper or cloth)\n"
                "- **Natural daylight**, no harsh shadows or flash\n"
                "- Hold the camera **parallel** to the leaf and keep it steady"
            )
            if consensus_votes:
                st.caption(
                    f"🗳️ Consensus: only {consensus_votes}/{consensus_total} crops agree"
                )
            result_area = st.expander(
                "🔓 Show the raw prediction anyway (unverified — do not act on it)",
                expanded=False,
            )

        with result_area:
            col_img, col_result = st.columns([1, 1.2])

            with col_img:
                if auto_crop:
                    st.image(original, caption=tr("original", lang), use_container_width=True)
                    st.image(processed, caption=tr("after_crop", lang), use_container_width=True)
                    if not crop_applied:
                        st.caption(tr("no_leaf", lang))
                else:
                    st.image(processed, caption=tr("uploaded_image", lang), use_container_width=True)

                # Quality chips
                cov_cls = "pv-chip" if not quality["poor_framing"] else "pv-chip pv-chip-warn"
                blur_cls = "pv-chip" if not quality["is_blurry"] else "pv-chip pv-chip-warn"
                st.markdown(
                    f"<div style='margin-top:.5rem'>"
                    f"<span class='{cov_cls}'>🍃 {tr('quality_leafcov', lang)}: {lesion['leaf_coverage']:.0f}%</span>"
                    f"<span class='{blur_cls}'>🔎 {tr('quality_blur', lang)}: {quality['sharpness']:.0f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if quality["poor_framing"]:
                    st.caption("⚠️ " + tr("quality_low", lang))
                if quality["is_blurry"]:
                    st.caption("⚠️ " + tr("quality_blurry", lang))

            # --- Result column ---
            with col_result:
                result_class = "pv-result-healthy" if is_healthy else "pv-result-diseased"
                icon = "✅" if is_healthy else "⚠️"
                condition_display = tr("healthy", lang) if is_healthy else condition

                st.markdown(f"""
                <div class="pv-result {result_class}">
                    <div class="pv-result-icon">{icon}</div>
                    <p class="pv-result-plant">{plant}</p>
                    <p class="pv-result-condition">{condition_display}</p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence gauge (Plotly)
                st.plotly_chart(
                    plotly_confidence_gauge(top_prob, is_healthy, tr("confidence_title", lang)),
                    use_container_width=True,
                )

                # Health Score gauge (SVG)
                st.markdown(f"#### 💚 {tr('health_score', lang)}")
                st.markdown(health_score_gauge(health_score, lang), unsafe_allow_html=True)

                # Low confidence warning
                if top_prob < 0.80:
                    st.warning(tr("low_conf", lang).format(pct=f"{top_prob:.1%}"))
                elif top_prob < 0.95:
                    st.info(tr("mod_conf", lang).format(pct=f"{top_prob:.1%}"))

            # ------------------------------------------------------------------
            # Prediction reliability (out-of-distribution guard)
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown(f"### {tr('reliab_title', lang)}")
            r1, r2 = st.columns(2)
            r1.metric(tr("reliab_entropy", lang), f"{reliability['entropy']:.3f}")
            r2.metric(tr("reliab_margin", lang), f"{reliability['margin']:.3f}")

            if reliability["is_ood"]:
                st.error("🛑 " + tr("reliab_ood", lang))
            elif reliability["is_close"]:
                st.warning("⚖️ " + tr("reliab_close", lang))
            else:
                st.success("✅ " + tr("reliab_ok", lang))

            # ------------------------------------------------------------------
            # Lesion area analysis
            # ------------------------------------------------------------------
            if show_lesions:
                st.markdown("---")
                st.markdown(f"### {tr('lesion_title', lang)}")
                st.caption(tr("lesion_caption", lang))

                l1, l2 = st.columns([1, 1])
                with l1:
                    st.image(
                        lesion["overlay"],
                        caption=tr("lesion_overlay", lang),
                        use_container_width=True,
                    )
                with l2:
                    band_label, band_color = lesion_band(affected_pct, lang)
                    st.plotly_chart(plotly_area_donut(affected_pct, lang), use_container_width=True)
                    st.markdown(
                        f"""
                        <div class="pv-glass">
                            <h4>{tr('lesion_affected', lang)}</h4>
                            <div style="display:flex;justify-content:space-between;align-items:baseline;">
                                <span style="color:#f0fdf4;font-size:1.8rem;font-weight:800;">{affected_pct:.1f}%</span>
                                <span class="pv-chip" style="color:{band_color};border-color:{band_color}55;">{band_label}</span>
                            </div>
                            <div class="pv-bar-track">
                                <div class="pv-bar-fill" style="width:{min(affected_pct,100):.1f}%;background:{band_color};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if affected_pct < 1.5:
                        st.caption(tr("lesion_none", lang))

            # --- Disease info ---
            info = DISEASE_INFO.get(top_class)
            if info:
                st.markdown("---")
                st.markdown(f"### {tr('about_result', lang)}")
                col_a, col_b = st.columns(2)

                with col_a:
                    sev_color = {
                        "None": "#4ade80",
                        "Moderate": "#fbbf24",
                        "Severe": "#f87171",
                    }.get(info["severity"], "#94a3b8")
                    st.markdown(f"""
                    <div class="pv-info-box">
                        <h4>{tr('condition', lang)}</h4>
                        <p><strong style="color:{sev_color};">{tr('severity', lang)}: {info['severity']}</strong></p>
                        <p style="margin-top:0.5rem;">{info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown(f"""
                    <div class="pv-info-box">
                        <h4>{tr('care_advice', lang)}</h4>
                        <p>{info['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # --- Top-5 candidates: chart + ranked list ---
            st.markdown("---")
            st.markdown(f"### {tr('top5_title', lang)}")
            st.plotly_chart(
                plotly_top5_bars(results, lang, AR_CLASS_NAMES),
                use_container_width=True,
            )

            for i, (raw_class, prob) in enumerate(results):
                p, c, healthy = parse_class(raw_class)
                if lang == "ar" and raw_class in AR_CLASS_NAMES:
                    p, c = AR_CLASS_NAMES[raw_class]
                label = f"🌱 {p} — {tr('healthy', lang)}" if healthy else f"🦠 {p} — {c}"
                rank_class = "pv-rank-item first" if i == 0 else "pv-rank-item"
                st.markdown(f"""
                <div class="{rank_class}">
                    <span class="pv-rank-name">#{i+1} · {label}</span>
                    <span class="pv-rank-prob">{prob:.2%}</span>
                </div>
                """, unsafe_allow_html=True)

            # --- Grad-CAM attention map ---
            if show_gradcam:
                st.markdown("---")
                st.markdown(f"### {tr('gradcam_title', lang)}")
                st.caption(tr("gradcam_caption", lang))

                with st.spinner(tr("gradcam_spinner", lang)):
                    try:
                        top_idx = [k for k, v in idx_to_class.items() if v == top_class][0]
                        heatmap = generate_gradcam(model, device, processed, target_class=top_idx)
                        st.image(
                            heatmap,
                            caption=f"{tr('gradcam_attn', lang)}: {plant} — {condition}",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"{tr('gradcam_fail', lang)}: {e}")

            # ------------------------------------------------------------------
            # Export + save to history
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown(f"### {tr('report_title', lang)}")
            st.caption(tr("report_caption", lang))

            overlay_b64 = ""
            if show_lesions:
                overlay_b64 = image_to_b64(Image.fromarray(lesion["overlay"]))

            report_html = build_report_html(
                lang=lang,
                plant=plant,
                condition=condition,
                is_healthy=is_healthy,
                confidence=top_prob,
                health_score=health_score,
                affected_pct=affected_pct,
                info=info,
                results=results,
                image_b64=image_to_b64(processed),
                overlay_b64=overlay_b64,
                reliability=reliability,
            )

            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    tr("report_btn", lang),
                    data=report_html.encode("utf-8"),
                    file_name=f"plantvision_report_{datetime.now():%Y%m%d_%H%M%S}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with e2:
                if st.button(tr("hist_save", lang), use_container_width=True):
                    save_scan(
                        plant=plant,
                        condition=tr("healthy", lang) if is_healthy else condition,
                        is_healthy=is_healthy,
                        confidence=top_prob,
                        health_score=health_score,
                        affected_pct=affected_pct,
                        raw_class=top_class,
                        pil_image=processed,
                    )
                    st.success(tr("hist_saved", lang))

            # --- Disclaimer ---
            st.markdown("---")
            st.caption(tr("disclaimer", lang))



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
    - Lesion area is measured from pixels, not from a trained segmentation model
    - Pest detection (YOLO) and the RAG recommendation engine are still planned
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


# ---------------------------------------------------------------------------
# Batch Scan tab — analyse many leaves at once
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown(f"## {tr('batch_title', lang)}")
    st.caption(tr("batch_caption", lang))

    batch_files = st.file_uploader(
        tr("batch_upload", lang),
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_files",
    )
    batch_crop = st.toggle(tr("autocrop_label", lang), value=True, key="batch_crop")

    if not batch_files:
        st.info(tr("batch_none", lang))
    else:
        if st.button(tr("batch_run", lang), type="primary"):
            progress = st.progress(0.0)
            rows = []
            for i, f in enumerate(batch_files):
                img = Image.open(f).convert("RGB")
                proc = auto_crop_leaf(img) if batch_crop else img
                res, pvec = predict_detailed(model, idx_to_class, device, proc, k=3)
                raw, prob = res[0]
                p_name, cond, healthy = parse_class(raw)
                les = analyze_lesions(proc)
                rel = reliability_stats(pvec)
                score = refine_health_score(
                    calculate_health_score(raw, prob), les["affected_pct"], healthy
                )
                rows.append({
                    "file": f.name,
                    "crop": p_name,
                    "condition": "Healthy" if healthy else cond,
                    "healthy": healthy,
                    "confidence": round(prob * 100, 2),
                    "health_score": score,
                    "affected_%": les["affected_pct"],
                    "uncertainty": rel["entropy"],
                    "flagged": bool(rel["is_ood"]),
                })
                progress.progress(
                    (i + 1) / len(batch_files),
                    text=f"{tr('batch_progress', lang)} {i + 1}/{len(batch_files)}",
                )
            progress.empty()
            st.session_state["batch_df"] = pd.DataFrame(rows)

        batch_df = st.session_state.get("batch_df")
        if batch_df is not None and not batch_df.empty:
            st.markdown("---")
            st.markdown(f"### {tr('batch_summary', lang)}")

            b1, b2, b3, b4 = st.columns(4)
            b1.metric(tr("hist_total", lang), len(batch_df))
            b2.metric(tr("batch_healthy_ratio", lang),
                      f"{100 * batch_df['healthy'].mean():.0f}%")
            b3.metric(tr("batch_avg_conf", lang),
                      f"{batch_df['confidence'].mean():.1f}%")
            b4.metric(tr("batch_avg_score", lang),
                      f"{batch_df['health_score'].mean():.0f}/100")

            c1, c2 = st.columns(2)
            with c1:
                status_counts = batch_df["healthy"].map(
                    {True: tr("hist_healthy", lang), False: tr("hist_diseased", lang)}
                ).value_counts()
                fig_status = go.Figure(go.Pie(
                    labels=status_counts.index.tolist(),
                    values=status_counts.values.tolist(),
                    hole=0.55,
                    marker=dict(colors=["#22c55e", "#f87171"]),
                ))
                fig_status.update_layout(
                    title=tr("batch_chart_status", lang), height=330,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", y=-0.1, bgcolor="rgba(0,0,0,0)"),
                    **PLOTLY_DARK,
                )
                st.plotly_chart(fig_status, use_container_width=True)

            with c2:
                crop_counts = batch_df["crop"].value_counts().reset_index()
                crop_counts.columns = ["crop", "count"]
                fig_crop = px.bar(
                    crop_counts, x="count", y="crop", orientation="h",
                    color="count", color_continuous_scale="Greens",
                )
                fig_crop.update_layout(
                    title=tr("batch_chart_crop", lang), height=330,
                    coloraxis_showscale=False, xaxis_title="", yaxis_title="",
                    margin=dict(l=10, r=10, t=50, b=10), **PLOTLY_DARK,
                )
                st.plotly_chart(fig_crop, use_container_width=True)

            st.markdown(f"### {tr('batch_table', lang)}")
            st.dataframe(batch_df, use_container_width=True, hide_index=True)

            st.download_button(
                tr("batch_csv", lang),
                data=batch_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"plantvision_batch_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# History tab — local SQLite scan log
# ---------------------------------------------------------------------------
with tab_history:
    st.markdown(f"## {tr('hist_title', lang)}")
    st.caption(tr("hist_caption", lang))

    hist_df = load_history()

    if hist_df.empty:
        st.info(tr("hist_empty", lang))
    else:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric(tr("hist_total", lang), len(hist_df))
        h2.metric(tr("hist_healthy", lang), int(hist_df["is_healthy"].sum()))
        h3.metric(tr("hist_diseased", lang), int((1 - hist_df["is_healthy"]).sum()))
        h4.metric(tr("hist_avg", lang), f"{hist_df['health_score'].mean():.0f}/100")

        # --- health score trend ---
        trend = hist_df.sort_values("id")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend["ts"], y=trend["health_score"],
            mode="lines+markers", name=tr("health_score", lang),
            line=dict(color="#4ade80", width=3),
            marker=dict(size=9, color=trend["health_score"],
                        colorscale=["#f87171", "#fbbf24", "#4ade80"],
                        cmin=0, cmax=100),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.10)",
        ))
        fig_trend.update_layout(
            title=tr("hist_trend", lang), height=340,
            yaxis_range=[0, 100], xaxis_title="", yaxis_title="",
            margin=dict(l=10, r=10, t=50, b=10), **PLOTLY_DARK,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # --- most scanned crops ---
        crops = hist_df["plant"].value_counts().reset_index()
        crops.columns = ["plant", "count"]
        fig_crops = px.bar(
            crops.head(10), x="count", y="plant", orientation="h",
            color="count", color_continuous_scale="Greens",
        )
        fig_crops.update_layout(
            title=tr("hist_dist", lang), height=330, coloraxis_showscale=False,
            xaxis_title="", yaxis_title="",
            margin=dict(l=10, r=10, t=50, b=10), **PLOTLY_DARK,
        )
        st.plotly_chart(fig_crops, use_container_width=True)

        # --- scan cards ---
        st.markdown("---")
        for _, row in hist_df.head(40).iterrows():
            score = int(row["health_score"])
            color = "#4ade80" if score >= 80 else "#fbbf24" if score >= 50 else "#f87171"
            thumb = (f'<img src="data:image/jpeg;base64,{row["thumb_b64"]}" alt="">'
                     if row["thumb_b64"] else "")
            st.markdown(f"""
            <div class="pv-timeline-item">
                {thumb}
                <div class="pv-timeline-main">
                    <div class="pv-timeline-title">{row['plant']} — {row['condition']}</div>
                    <div class="pv-timeline-sub">
                        🕒 {row['ts']} · 🎯 {row['confidence']:.1%} · 🔬 {row['affected_pct']:.1f}%
                    </div>
                </div>
                <div class="pv-timeline-score" style="color:{color};">{score}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.download_button(
                tr("hist_csv", lang),
                data=hist_df.drop(columns=["thumb_b64"]).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"plantvision_history_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with hc2:
            if st.button(tr("hist_clear", lang), use_container_width=True):
                clear_history()
                st.success(tr("hist_cleared", lang))
                st.rerun()


# ---------------------------------------------------------------------------
# Knowledge Base tab — searchable agricultural notes
# ---------------------------------------------------------------------------
with tab_kb:
    st.markdown(f"## {tr('kb_title', lang)}")
    st.caption(tr("kb_caption", lang))

    all_crops = sorted({parse_class(c)[0] for c in DISEASE_INFO})

    k1, k2 = st.columns([2, 1])
    with k1:
        query = st.text_input(
            tr("kb_search", lang),
            placeholder=tr("kb_placeholder", lang),
        )
    with k2:
        crop_filter = st.selectbox(
            tr("kb_filter", lang),
            options=[tr("kb_all", lang)] + all_crops,
        )

    if query.strip():
        hits = search_kb(query, top_k=8)
    else:
        hits = [(raw, 0.0) for raw in DISEASE_INFO]

    if crop_filter != tr("kb_all", lang):
        hits = [(raw, s) for raw, s in hits if parse_class(raw)[0] == crop_filter]

    if not hits:
        st.warning(tr("kb_no_results", lang))
    else:
        st.caption(f"**{len(hits)}** {tr('kb_results', lang)}")
        for raw, score in hits:
            info = DISEASE_INFO[raw]
            p_name, cond, healthy = parse_class(raw)
            disp_p, disp_c = p_name, cond
            if lang == "ar" and raw in AR_CLASS_NAMES:
                disp_p, disp_c = AR_CLASS_NAMES[raw]
            sev = info.get("severity", "—")
            sev_chip = {
                "None": "pv-chip",
                "Moderate": "pv-chip pv-chip-warn",
                "Severe": "pv-chip pv-chip-danger",
            }.get(sev, "pv-chip")
            rel_chip = (f"<span class='pv-chip pv-chip-info'>{tr('kb_relevance', lang)}: "
                        f"{score:.0%}</span>" if score > 0 else "")
            icon = "🌱" if healthy else "🦠"
            st.markdown(f"""
            <div class="pv-kb-card">
                <h5>{icon} {disp_p} — {disp_c}</h5>
                <div>
                    <span class="{sev_chip}">{tr('kb_severity', lang)}: {sev}</span>
                    {rel_chip}
                </div>
                <p>{info.get('description', '')}</p>
                <p><strong style="color:#86efac;">{tr('kb_advice', lang)}:</strong>
                   {info.get('advice', '')}</p>
            </div>
            """, unsafe_allow_html=True)


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