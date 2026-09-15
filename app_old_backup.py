"""
PlantVision AI — Streamlit Application (Phase 10 preview)
Run from the project root:  streamlit run app.py
"""

from __future__ import annotations

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

sys.path.append(str(Path(__file__).resolve().parent))
from src.classification.plant_classifier import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "plant_classifier" / "best_model.pt"
METRICS_PATH = ROOT / "reports" / "plant_classifier_metrics.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Model loading ----------
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


def parse_class(raw: str) -> tuple[str, str, bool]:
    if "___" in raw:
        plant, condition = raw.split("___", 1)
    else:
        plant, condition = raw, "unknown"
    plant = plant.replace("_", " ").strip()
    condition_clean = condition.replace("_", " ").strip()
    is_healthy = condition.lower() == "healthy"
    return plant, condition_clean, is_healthy


def predict(model, idx_to_class, device, image, k: int = 5):
    tf = build_transform()
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu()
    top_probs, top_idx = torch.topk(probs, k=min(k, len(idx_to_class)))
    return [(idx_to_class[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]


# ---------- Auto-crop ----------
def auto_crop_leaf(pil_image: Image.Image, margin: int = 10) -> Image.Image:
    """
    Detect the main leaf in an image and crop to its bounding box.

    Uses GrabCut on a downsampled version for speed, then scales the
    resulting bounding box back to the original resolution.

    Falls back to the original image if detection fails.
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    # Work at reduced size for speed (GrabCut is slow on big images)
    max_side = 512
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img

    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), dtype=np.uint8)

    # Initial rectangle: 5% inset from each edge
    rect = (int(sw * 0.05), int(sh * 0.05),
            int(sw * 0.90), int(sh * 0.90))

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return pil_image  # fallback

    # Foreground = definite + probable
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")

    # Find largest connected foreground component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, 8)
    if num_labels <= 1:
        return pil_image  # nothing found

    # Skip background label (0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x, y, bw, bh = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP], \
                   stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]

    # Filter out tiny detections (likely noise)
    if bw * bh < 0.05 * sw * sh:
        return pil_image

    # Scale box back to original size + add margin
    inv = 1.0 / scale
    x0 = max(0, int(x * inv) - margin)
    y0 = max(0, int(y * inv) - margin)
    x1 = min(w, int((x + bw) * inv) + margin)
    y1 = min(h, int((y + bh) * inv) + margin)

    return pil_image.crop((x0, y0, x1, y1))


# ---------- Load model and metrics ----------
model, idx_to_class, device, ckpt = load_model()
metrics = load_metrics()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌿 PlantVision AI")
    st.caption("Preview build — plant classifier only")
    st.divider()

    st.markdown("### Model info")
    st.write(f"**Device:** `{device}`")
    st.write("**Architecture:** MobileNetV2")
    st.write(f"**Classes:** {len(idx_to_class)}")
    st.write(f"**Trained epochs:** {ckpt['epoch']}")
    st.write(f"**Val accuracy:** {ckpt['val_acc']:.2%}")

    if metrics:
        st.markdown("### Test set performance")
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Top-5 accuracy", f"{metrics['top5_accuracy']:.2%}")
        st.metric("F1 (macro)", f"{metrics['f1_macro']:.2%}")
        st.caption(f"on {metrics['num_test_images']:,} held-out images")

    st.divider()
    st.caption(
        "Preview only. Pest detection, severity, health score, and recommendations "
        "are planned for later phases."
    )

# ---------- Header ----------
st.title("🌿 PlantVision AI")
st.markdown(
    "**From Plant Images to Agricultural Intelligence.** "
    "Upload a leaf photo and the model will identify the plant and its health state."
)

# ---------- Tabs ----------
tab_predict, tab_info = st.tabs(["🔍 Predict", "📊 Model info"])

with tab_predict:
    with st.expander("📸 How to take a good photo (important for accuracy)", expanded=False):
        st.markdown(
            """
            The model was trained on **PlantVillage-style** images: a single leaf, plain
            background, even lighting. To get the best predictions:

            ✅ **DO:**
            - Photograph **one leaf only** — fill the frame with it
            - Place it on a **plain background** (white paper, cloth, floor)
            - Use **natural daylight** or bright, even indoor lighting
            - Hold the camera **parallel** to the leaf (top-down view)

            ❌ **DON'T:**
            - Photograph the whole plant or a full branch
            - Use busy backgrounds (soil, grass, other plants)
            - Use flash or harsh shadows
            - Zoom out — get close to the leaf

            💡 **Tip:** Enable "Auto-crop leaf" below to let the app isolate
            the leaf automatically before prediction.
            """
        )

    auto_crop = st.toggle(
        "✂️ Auto-crop leaf (recommended)",
        value=True,
        help="Automatically detect and isolate the main leaf before prediction."
    )

    uploaded = st.file_uploader(
        "Upload a plant leaf image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("👆 Upload an image to see a prediction.")
    else:
        original = Image.open(uploaded).convert("RGB")

        # Apply auto-crop if enabled
        if auto_crop:
            with st.spinner("Detecting leaf..."):
                cropped = auto_crop_leaf(original)
            display_img = cropped
            cropped_ok = cropped.size != original.size
        else:
            display_img = original
            cropped_ok = None

        # Low-resolution check
        w, h = display_img.size
        if min(w, h) < 200:
            st.warning(
                f"⚠️ Image is small ({w}×{h}). Predictions may be unreliable."
            )

        col_img, col_res = st.columns([1, 1.3])

        with col_img:
            if auto_crop:
                st.image(original, caption="Original", use_container_width=True)
                st.image(display_img, caption="After auto-crop", use_container_width=True)
                if not cropped_ok:
                    st.caption("_(No leaf detected — using original image)_")
            else:
                st.image(display_img, caption="Uploaded image", use_container_width=True)

        with col_res:
            with st.spinner("Analyzing..."):
                results = predict(model, idx_to_class, device, display_img)

            top_class, top_prob = results[0]
            plant, condition, is_healthy = parse_class(top_class)

            if is_healthy:
                st.success(f"### ✅ {plant}\n**Healthy**")
            else:
                st.error(f"### ⚠️ {plant}\n**Diseased — {condition}**")

            st.metric("Confidence", f"{top_prob:.2%}")

            if top_prob < 0.80:
                st.warning(
                    f"⚠️ **Low confidence ({top_prob:.1%}).** The model is uncertain — "
                    f"this image may differ from the training data. Try auto-crop, "
                    f"or retake the photo following the tips above."
                )
            elif top_prob < 0.95:
                st.info(
                    f"ℹ️ Moderate confidence ({top_prob:.1%}). "
                    f"Consider retaking the photo for a more reliable result."
                )

            st.markdown("#### Top 5 candidates")
            for raw_class, prob in results:
                p, c, h = parse_class(raw_class)
                label = f"🌱 {p} — **Healthy**" if h else f"🦠 {p} — {c}"
                st.write(label)
                st.progress(min(prob, 1.0), text=f"{prob:.2%}")

        if not is_healthy:
            st.warning(
                "⚠️ Raw classifier output for one image — not a full plant health report. "
                "Confirm with a professional before any treatment."
            )
        else:
            st.info("🌱 The model predicts this leaf is healthy. Keep monitoring for early signs.")

with tab_info:
    st.subheader("Model details")
    st.markdown(
        """
        **Architecture:** MobileNetV2 (transfer learning from ImageNet)

        **Training data:** PlantVillage — 38 classes (14 crops × healthy/diseased)

        **Training setup:**
        - Backbone frozen (head-only training)
        - Adam, lr=1e-3
        - 15 epochs (best checkpoint at epoch 14)
        - WeightedRandomSampler for class imbalance
        - Mixed-precision (AMP) on GPU

        **Auto-crop:** GrabCut (OpenCV) — isolates the main leaf before inference.
        """
    )

    if metrics:
        st.subheader("Test set metrics (held-out)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Top-5 accuracy", f"{metrics['top5_accuracy']:.2%}")
        c3.metric("F1 (macro)", f"{metrics['f1_macro']:.2%}")
        st.caption(
            f"Evaluated on **{metrics['num_test_images']:,}** images never seen "
            f"during training or validation. All numbers come from "
            f"`reports/plant_classifier_metrics.json` — none are hand-written."
        )

        st.subheader("Classes the model can recognize")
        crops = sorted({parse_class(c)[0] for c in idx_to_class.values()})
        st.write(f"**{len(crops)} crops:** " + ", ".join(crops))

        st.subheader("Worst 5 classes (by F1)")
        for row in metrics.get("worst_5_classes", []):
            plant, cond, _ = parse_class(row["class"])
            st.write(f"- {plant} — {cond}: F1 = {row['f1']:.2%}")
    else:
        st.warning("No metrics file. Run `python training/evaluate.py` first.")

    st.divider()
    st.caption(
        "**Limitations:** Works best on clear, single-leaf photos with plain "
        "background (like PlantVillage). Real field photos may reduce accuracy. "
        "Pest detection and severity are separate modules for later phases."
    )