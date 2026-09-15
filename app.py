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


# ============================================================================
# Load model and metrics
# ============================================================================
model, idx_to_class, device, ckpt = load_model()
metrics = load_metrics()


# ============================================================================
# Header — Hero section
# ============================================================================
st.markdown(f"""
<div class="pv-hero">
    <h1>🌿 PlantVision AI</h1>
    <p>{tr('hero_tagline', lang)}</p>
    <p style="font-size:0.95rem;opacity:0.8;margin-top:0.5rem;">
        {tr('hero_subtitle', lang)}
    </p>
    <span class="pv-badge">{tr('hero_badge', lang)}</span>
</div>
""", unsafe_allow_html=True)


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
tab_predict, tab_info, tab_analytics = st.tabs([
    tr("tab_predict", lang),
    tr("tab_info", lang),
    tr("tab_analytics", lang),
])


# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    # --- Upload section ---
    col_upload_left, col_upload_right = st.columns([2, 1])

    with col_upload_left:
        uploaded = st.file_uploader(
            tr("upload_label", lang),
            type=["jpg", "jpeg", "png"],
            help=tr("upload_help", lang),
        )

    with col_upload_right:
        auto_crop = st.toggle(
            tr("autocrop_label", lang),
            value=True,
            help=tr("autocrop_help", lang),
        )
        st.caption(tr("autocrop_caption", lang))

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

    # --- Result section ---
    if uploaded is None:
        st.info(tr("no_image", lang))
    else:
        original = Image.open(uploaded).convert("RGB")

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

        # --- Run prediction ---
        with st.spinner(tr("analyzing", lang)):
            results = predict(model, idx_to_class, device, processed)

        top_class, top_prob = results[0]
        plant, condition, is_healthy = parse_class(top_class)

        # Arabic name override
        if lang == "ar" and top_class in AR_CLASS_NAMES:
            plant_ar, condition_ar = AR_CLASS_NAMES[top_class]
            plant = plant_ar
            condition = condition_ar

        # --- Health score ---
        health_score = calculate_health_score(top_class, top_prob)

        # --- Image column ---
        col_img, col_result = st.columns([1, 1.2])

        with col_img:
            if auto_crop:
                st.image(original, caption=tr("original", lang), use_container_width=True)
                st.image(processed, caption=tr("after_crop", lang), use_container_width=True)
                if not crop_applied:
                    st.caption(tr("no_leaf", lang))
            else:
                st.image(processed, caption=tr("uploaded_image", lang), use_container_width=True)

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

            st.markdown("---")

            # Confidence gauge
            st.markdown(confidence_gauge(top_prob, is_healthy), unsafe_allow_html=True)

            # Health Score gauge
            st.markdown(f"#### 💚 {tr('health_score', lang)}")
            st.markdown(health_score_gauge(health_score, lang), unsafe_allow_html=True)

            # Low confidence warning
            if top_prob < 0.80:
                st.warning(tr("low_conf", lang).format(pct=f"{top_prob:.1%}"))
            elif top_prob < 0.95:
                st.info(tr("mod_conf", lang).format(pct=f"{top_prob:.1%}"))

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

        # --- Top-5 candidates ---
        st.markdown("---")
        st.markdown(f"### {tr('top5_title', lang)}")

        for i, (raw_class, prob) in enumerate(results):
            p, c, healthy = parse_class(raw_class)
            if lang == "ar" and raw_class in AR_CLASS_NAMES:
                p_ar, c_ar = AR_CLASS_NAMES[raw_class]
                p = p_ar
                c = c_ar
            label = f"🌱 {p} — {tr('healthy', lang)}" if healthy else f"🦠 {p} — {c}"
            rank_class = "pv-rank-item first" if i == 0 else "pv-rank-item"
            st.markdown(f"""
            <div class="{rank_class}">
                <span class="pv-rank-name">#{i+1} · {label}</span>
                <span class="pv-rank-prob">{prob:.2%}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- Grad-CAM attention map ---
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