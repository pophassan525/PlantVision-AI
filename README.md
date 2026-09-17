# 🌿 PlantVision AI

> **From Plant Images to Agricultural Intelligence**

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff0000?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Phase%202%20%2B%20Guards-brightgreen?style=flat-square)

---

## 🎯 Overview

**PlantVision AI** is an intelligent plant health analysis platform that transforms a single leaf photo into a comprehensive health report. Using computer vision and machine learning, it identifies the plant, detects diseases, measures affected tissue area, and provides actionable care advice — all in seconds.

The system is **designed for real-world use**: it knows its limitations, refuses uncertain predictions, and guides users toward better photos when needed.

---

## 🔬 What It Does

| Input | Process | Output |
|-------|---------|--------|
| 📸 Leaf photo | **MobileNetV2 classifier** trained on 38 crop-disease classes | ✅ Plant ID + disease status |
| | **Auto-crop** with OpenCV (HSV + GrabCut) | 📐 Clean leaf region |
| | **Lesion analysis** (pixel-level measurement) | 📊 % affected tissue area |
| | **Quality check** (sharpness + framing) | ⚠️ Photo reliability warnings |
| | **Confidence gates** (entropy + margin guards) | 🛡️ Safe/unsafe prediction flag |
| → | **Grad-CAM** attention map | 🔍 Where the model looked |
| | **Batch scan** for multiple leaves | 📦 Field-scale results + CSV |
| | **Scan history** (SQLite database) | 🕒 Health trends over time |
| → | **Knowledge base** search | 📚 Searchable symptom/treatment database |

---

## ✨ Key Features

### **Core Classifier**
- **94.1% test accuracy** on PlantVillage (38 classes, 14 crops)
- **MobileNetV2** transfer learning (lightweight, fast inference)
- **Supports:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### **Image Processing**
- ✂️ **Auto-crop** with HSV masking + GrabCut refinement
- 📊 **Histogram normalization** (CLAHE) for field photos with uneven lighting
- 🔍 **Lesion area measurement** — real pixel-level analysis, not guesses
- 📐 **Quality metrics** — sharpness (Laplacian) + leaf coverage check

### **Confidence Guards** (Blockers vs Warnings)
| Level | Trigger | Outcome |
|-------|---------|---------|
| **BLOCKER** | Confidence < per-class threshold | ❌ Prediction hidden by default |
| | Entropy > 0.40 (model uncertain) | Shows in expander: "unverified" |
| | Top-1/Top-2 margin < 0.12 (tied) | |
| **WARNING** | Leaf < 15% of frame (too small) | ⚠️ Shown in the result |
| | Sharpness < 60 (blurry) | Doesn't block unless 2+ warnings |

*Per-class thresholds adapt based on model F1 scores — easier classes (F1 > 0.95) need 70% confidence; harder ones (F1 < 0.85) need 86%.*

### **Reports & Export**
- 📄 **HTML report** (self-contained, prints to PDF)
- 📦 **Batch CSV** with per-image scores
- 🕒 **History export** — track plant health over time

### **UI / UX**
- 🌓 **Bilingual** (English + العربية with full RTL support)
- 🎨 **Aurora animations** + glassmorphic design
- 📱 **Responsive** (desktop, tablet, mobile)
- 🎯 **Consensus voting** (optional) — compares 5 image crops for extra confidence

---

## 📊 Model Performance

**Test Set Metrics (held-out, n=1,245 images):**

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.1% |
| **Top-5 Accuracy** | 99.7% |
| **F1 (macro)** | 91.2% |
| **F1 (weighted)** | 94.1% |

**Per-class F1 range:** 72% (Tomato late blight) → 99% (Potato healthy)  
**Worst 5 classes:** tomato late blight (72%), potato late blight (74%), apple scab (76%), grape black rot (77%), corn rust (79%)

---

## 🚀 Quick Start

### **Installation**

```bash
# Clone the repo
git clone https://github.com/pophassan525/PlantVision-AI.git
cd PlantVision-AI

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy config
cp .env.example .env
```

### **Run the App**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📖 How to Use

### **Single Leaf Analysis**
1. **Upload** a photo of a single leaf (JPG/PNG)
   - Best: plain background, natural daylight, leaf fills frame
2. **Toggle** auto-crop (recommended for field photos)
3. **View results:**
   - Plant ID + disease (if any)
   - Confidence gauge
   - Health score (0–100)
   - Lesion area % (affected tissue)
   - Reliability warning (if image quality is poor)
   - Top 5 candidates
   - Model attention map (Grad-CAM)
4. **Export** as HTML report or save to history

### **Batch Scan**
Upload 2+ images, click "Analyse all images" → get summary stats, per-image table, CSV download.

### **View History**
See all saved scans, health trends over time, most-scanned crops.

### **Knowledge Base**
Search by symptom, disease, or crop — view full descriptions and treatment advice for all 38 classes.

---

## ⚙️ Technology Stack

### **Implemented**
- **Python 3.11** with type hints
- **PyTorch 2.0+** (MobileNetV2, Grad-CAM)
- **OpenCV 4.8+** (auto-crop, lesion detection, quality metrics)
- **Streamlit 1.28+** (interactive UI, bilingual)
- **Scikit-learn** (metrics, TF-IDF search)
- **Plotly** (interactive charts)
- **SQLite** (scan history)
- **Pandas + NumPy** (data processing)

### **Planned**
- **Ultralytics YOLO** (pest detection — Phase 5)
- **LangChain + FAISS** (RAG assistant — Phase 9)
- **PostgreSQL** (cloud history)

---

## 📁 Project Structure

```
PlantVision-AI/
├── app.py                          # Main Streamlit app (3,195 lines)
├── requirements.txt                # Python dependencies
├── .env.example                    # Config template
├── .streamlit/config.toml          # Streamlit settings
├── models/
│   └── plant_classifier/
│       └── best_model.pt           # MobileNetV2 checkpoint
├── data/
│   ├── train/                      # Training split
│   ├── validation/                 # Validation split
│   ├── test/                       # Test split (for "Sample leaf")
│   └── plantvision_history.db      # SQLite scan log (auto-created)
├── reports/
│   ├── plant_classifier_metrics.json
│   ├── plant_classifier_confusion_matrix.png
│   └── plant_classifier_training_log.csv
├── scripts/
│   ├── split_dataset.py            # Train/val/test split
│   ├── train_plant_classifier.py   # Training pipeline
│   └── evaluate.py                 # Metrics computation
├── src/
│   └── classification/
│       └── plant_classifier.py     # Model architecture
└── docs/
    ├── ROADMAP.md                  # Development phases
    ├── ARCHITECTURE.md             # System design
    ├── LIMITATIONS.md              # Honest capabilities
    └── DATASET_PLAN.md             # Data overview
```

---

## 🔐 Safety & Limitations

### **What It Can Do**
✅ Identify plants from single leaves (38 classes)  
✅ Detect visual disease symptoms  
✅ Measure lesion area from pixels  
✅ Flag low-confidence predictions  
✅ Provide treatment advice (from knowledge base)  

### **What It Cannot Do**
❌ Identify multi-part plant structures (whole plant, branches, roots)  
❌ Work on field photos with multiple overlapping leaves  
❌ Diagnose soil or nutrient deficiencies  
❌ Predict pest infestations (planned for Phase 5)  
❌ Replace professional agricultural diagnosis  
❌ Work on heavily filtered or edited photos  

### **Why Predictions Can Fail**
- Photo quality (blurry, too dark, strange angle)
- Plant outside the 38 supported classes
- Multiple leaves in one frame (auto-crop may pick the wrong one)
- Very unusual growing conditions that differ from training data
- Lighting that masks symptoms (shadows, glare)

**The app warns you about all of these.** If a prediction is uncertain, it says so.

---

## 📈 Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **0** | Project scaffold, config, docs | ✅ Done |
| **1** | Dataset pipeline (PlantVillage) | ✅ Done |
| **2** | Plant classifier + Streamlit UI | ✅ Done (94.1% accuracy) |
| **3** | Confidence guards + lesion analysis | ✅ Done |
| **4** | Batch scan + history database | ✅ Done |
| **5** | YOLO pest detection | 📋 Planned |
| **6** | Severity estimation (rule-based) | 📋 Planned |
| **7** | Knowledge base search (TF-IDF) | ✅ Done |
| **8** | Report generation (HTML/PDF) | ✅ Done |
| **9** | RAG assistant (LangChain + LLM) | 📋 Planned |

See `docs/ROADMAP.md` for detailed phase descriptions.

---

## 💡 Usage Tips

### **For Best Results**
1. **Photograph one healthy-looking leaf** in good condition
2. **Plain background** — white paper, cloth, or solid color (not wood/soil)
3. **Natural daylight** — avoid harsh shadows and flash
4. **Hold steady** — use a tripod for sharp focus
5. **Fill the frame** — the leaf should take up at least 15% of the image
6. **Top-down angle** — hold camera parallel to the leaf surface

### **If Prediction Fails**
- The app will tell you why (blurry, too small, uncertain model)
- Retake the photo with the tips above
- Toggle **auto-crop off** if the leaf is oddly shaped
- Check the **Top 5 candidates** — one of them might be correct
- Use **Consensus voting** for extra confidence (slower, but more reliable)

---

## 🤝 Contributing

Contributions are welcome! Areas we need help with:

- **Pest detection dataset** curation (Phase 5)
- **Localization** to more languages
- **Severity estimation** rules (agronomists welcome)
- **Performance optimization** for mobile devices
- **Unit tests** for the CV pipeline
- **Documentation** improvements

Please open an issue or PR on GitHub.

---

## 📜 License

MIT License — see `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** — Penn State, Image Analytics Laboratory
- **PyTorch + Torchvision** — Meta AI
- **MobileNetV2** — Google
- **Streamlit** — Streamlit Inc.
- **OpenCV** — OpenCV team

---

## 📧 Contact & Support

- **GitHub Issues:** https://github.com/pophassan525/PlantVision-AI/issues
- **Email:** [your email]

---

<div align="center">

**Made with 🌱 for farmers, gardeners, students, and plant lovers.**

</div>
