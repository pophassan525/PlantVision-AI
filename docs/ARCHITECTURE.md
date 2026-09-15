# PlantVision AI — Architecture

Status: Phase 0 — architecture defined, no modules implemented yet.

## 1. Vision

One image in → the maximum amount of scientifically defensible plant-health
intelligence out, clearly separating what the image can support from what it
cannot (soil chemistry, lab measurements, etc.).

## 2. High-Level Data Flow

```
IMAGE
  -> PREPROCESSING
  -> PLANT IDENTIFICATION
  -> DISEASE ANALYSIS
  -> PEST ANALYSIS (if model available)
  -> SYMPTOM ANALYSIS (model-detected + rule/knowledge-based, kept distinct)
  -> SEVERITY ESTIMATION
  -> PLANT HEALTH SCORE (config-driven, transparent weighting)
  -> AGRICULTURAL KNOWLEDGE RETRIEVAL (RAG)
  -> AI RECOMMENDATION
  -> FINAL PLANT HEALTH REPORT
  -> HISTORY STORAGE (SQLite)
```

## 3. Module Status Legend

Every module below is tagged with one of:
`IMPLEMENTED` / `IN DEVELOPMENT` / `PLANNED` / `FUTURE EXTENSION`.
The Streamlit UI and README must mirror these tags exactly — no module is ever
presented as more complete than this table says.

| Module | Status |
|---|---|
| Project scaffold / config | IMPLEMENTED |
| Dataset inspection & preprocessing pipeline | PLANNED (Phase 1) |
| Plant identification classifier | PLANNED (Phase 2) |
| Disease detection classifier | PLANNED (Phase 2/3) |
| Symptom analysis (rule-based) | PLANNED (Phase 3) |
| Pest detection (YOLO) | PLANNED (Phase 4) — depends on a bbox-labeled dataset being confirmed |
| Severity estimation | PLANNED (Phase 5) |
| Health score engine | PLANNED (Phase 6) — config file exists (`config/health_score.yaml`), engine not yet coded |
| Explainability (Grad-CAM etc.) | PLANNED (Phase 11) |
| Agricultural knowledge base | PLANNED (Phase 7) |
| RAG assistant | PLANNED (Phase 9) |
| Recommendation engine | PLANNED (Phase 10... per numbering, Module 10) |
| Plant history / SQLite storage | PLANNED (Phase 1 schema, wired in later) |
| Early warning (environmental+historical) | FUTURE EXTENSION |
| Streamlit application | PLANNED (Phase 10) |

## 4. Module Responsibilities (`src/`)

- `preprocessing/` — image validation, resizing, normalization, augmentation, corrupted/duplicate detection
- `classification/` — plant ID + disease classifiers (transfer learning wrappers)
- `detection/` — pest detector (YOLO)
- `segmentation/` — lesion/affected-area segmentation, when a labeled dataset supports it
- `severity/` — severity estimation logic, consuming segmentation/detection output
- `scoring/` — health score engine, reads `config/health_score.yaml`, never hard-codes weights
- `rag/` — retriever + LLM orchestration over `data/knowledge/`
- `recommendations/` — turns findings + RAG context into a structured recommendation, grounded in the knowledge base only
- `database/` — SQLite access layer for `plants`, `analyses`, `predictions`, `symptoms`, `recommendations`, `users`
- `utils/` — shared helpers (config loading, logging, device selection)

## 5. Hardware Assumption

No GPU is available in the current development environment. All model choices
default to CPU-feasible architectures (MobileNetV2 / EfficientNet-B0). GPU is
used automatically if `config/config.yaml: hardware.device: auto` detects one.

## 6. Explicit Non-Goals (Current Version)

The system does not and will not claim to determine, from an image alone:
soil pH, exact NPK concentration, exact soil moisture, root health, lab-grade
pathogen identification, or exact treatment dosages. See `docs/LIMITATIONS.md`.
