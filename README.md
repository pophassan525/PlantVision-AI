# PlantVision AI

**From Plant Images to Agricultural Intelligence.**

> ⚠️ Current status: **Phase 0 — architecture and scaffolding only.** No model
> has been trained, no dataset has been finalized, and no analysis pipeline is
> functional yet. Nothing in this README should be read as a description of a
> working feature unless explicitly marked IMPLEMENTED below.

## Overview

PlantVision AI is a modular, AI-powered plant health analysis platform. It is
designed to take a single plant image and extract the maximum amount of
scientifically defensible information about that plant's health — plant
identification, disease indicators, visible symptoms, pests (where a model
supports it), estimated severity, a transparent health score, and
knowledge-grounded recommendations.

## Problem

Small farmers, students, and plant owners often lack fast, low-friction access
to a first-pass assessment of plant health. Professional diagnosis is
valuable but not always immediately available.

## Solution

An image-in, report-out decision-support tool that is explicit about what it
can and cannot determine from a photo alone (see `docs/LIMITATIONS.md`), and
that never presents unvalidated or fabricated results as real.

## Architecture

See `docs/ARCHITECTURE.md` for the full module breakdown and data flow.

## Feature Status

| Feature | Status |
|---|---|
| Project scaffold, config, docs | **IMPLEMENTED** |
| Dataset pipeline | PLANNED |
| Plant identification | PLANNED |
| Disease detection | PLANNED |
| Symptom analysis | PLANNED |
| Pest detection | PLANNED |
| Severity estimation | PLANNED |
| Health score engine | PLANNED (config exists, engine does not) |
| Explainability (Grad-CAM) | PLANNED |
| Knowledge base | PLANNED |
| RAG assistant | PLANNED |
| Recommendation engine | PLANNED |
| Streamlit app | PLANNED |
| Plant history | PLANNED |
| Early warning / environmental integration | FUTURE EXTENSION |

## Technology Stack (planned)

- Python 3.12
- PyTorch/TorchVision (transfer learning: MobileNetV2 / EfficientNet-B0)
- Ultralytics YOLO (pest detection, once data supports it)
- Streamlit (application UI)
- SQLite (storage)
- RAG stack (vector store + LLM) — chosen in Phase 9

## Dataset

Not finalized. See `docs/DATASET_PLAN.md` — currently blocked on selecting and
inspecting a real, licensed dataset (candidates: PlantVillage, PlantDoc, IP102).

## Installation

```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Setup

Edit `.env` after copying from `.env.example`. No secrets should ever be
committed to version control.

## Training

Not available yet — depends on Phase 1 (dataset) and Phase 2 (baseline model).
Planned commands (not yet functional):

```bash
python training/train_plant_classifier.py
python training/train_disease_classifier.py
python training/evaluate.py
```

## Evaluation

Planned: `reports/` will contain `classification_report.txt`,
`confusion_matrix.png`, `metrics.json`, `evaluation_summary.md` — generated
from real runs only, never hand-written.

## Running the Application

Planned (Phase 10, not yet functional):

```bash
python -m streamlit run app.py
```

## Project Structure

```
plantvision_ai/
├── app.py                  # not yet created — Phase 10
├── config/
│   ├── config.yaml
│   └── health_score.yaml
├── data/
│   ├── raw/ processed/ train/ validation/ test/ knowledge/
├── models/
│   ├── plant_classifier/ disease_classifier/ pest_detector/ severity/
├── src/
│   ├── preprocessing/ classification/ detection/ segmentation/
│   ├── severity/ scoring/ rag/ recommendations/ database/ utils/
├── training/
├── scripts/
├── tests/
├── reports/
├── notebooks/
├── assets/
└── docs/
    ├── ARCHITECTURE.md
    ├── DATASET_PLAN.md
    ├── MODEL_PLAN.md
    ├── API_PLAN.md
    └── LIMITATIONS.md
```

## Limitations

See `docs/LIMITATIONS.md`.

## Future Development

See the roadmap (Versions 2–5) in `docs/ARCHITECTURE.md` — environmental data
integration, IoT sensor data, predictive agriculture, and field-level
monitoring. None of these are implemented in the current version.

## Ethical / Scientific Considerations

This system does not replace professional agricultural diagnosis. It does not
fabricate confidence values, accuracy figures, or disease predictions — any
module without a trained, evaluated model reports itself as unavailable
rather than guessing.

## Disclaimer

PlantVision AI provides AI-assisted analysis based on available image and
data inputs. Results should not replace professional agricultural diagnosis.
