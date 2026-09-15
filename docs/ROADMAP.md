# PlantVision AI — Roadmap (52 Weeks)

Status: PLANNED. This roadmap sequences the phases already listed in
`ARCHITECTURE.md` into weekly milestones. Dates are relative ("Week N"), not
calendar-locked — shift everything if a week runs long. Each week ends with
one concrete, demoable artifact (a script, a trained model, a report, a
working screen) so progress is visible even in a slow week.

**Rule carried over from the rest of this project:** no week's write-up
claims a result that wasn't actually produced by running code. If a week
slips, move the date, don't invent the output.

\---

## Month 1 — Dataset Foundation (Phase 1)

|Week|Deliverable|
|-|-|
|1|Dataset source decided (PlantVillage / PlantDoc / IP102 or a mirror) + license checked + raw files in `data/raw/`. Fill in `docs/DATASET\_PLAN.md`.|
|2|`scripts/check\_dataset.py` — runs the inspection checklist from `DATASET\_PLAN.md` (class counts, corrupted files, duplicates). Output: `reports/dataset\_inspection.md`.|
|3|`src/preprocessing/` implemented — validation, resizing, normalization, augmentation. Unit tests in `tests/`.|
|4|Train/validation/test split script + class-imbalance analysis + first exploratory notebook in `notebooks/` with sample images per class.|

## Month 2 — Plant Identification Model (Phase 2)

|Week|Deliverable|
|-|-|
|5|`src/classification/plant\_classifier.py` — model definition (MobileNetV2 or EfficientNet-B0 transfer learning), confirmed in `MODEL\_PLAN.md`.|
|6|`training/train\_plant\_classifier.py` runs end-to-end on real data; first trained checkpoint in `models/plant\_classifier/`.|
|7|`training/evaluate.py` — real accuracy/F1/confusion matrix in `reports/`.|
|8|Hyperparameter pass (learning rate, augmentation, epochs) + updated report comparing runs.|

## Month 3 — Disease Detection (Phase 2/3)

|Week|Deliverable|
|-|-|
|9|Disease-labeled subset confirmed and split; `disease\_classifier` model defined.|
|10|Disease classifier trained; checkpoint in `models/disease\_classifier/`.|
|11|Evaluation report for disease classifier (real metrics, confusion matrix per disease class).|
|12|`src/severity` groundwork stub wired to disease output (no logic yet) + `src/classification` API matches `API\_PLAN.md` contract.|

## Month 4 — Symptom Analysis + Explainability Preview (Phase 3, part of Phase 11)

|Week|Deliverable|
|-|-|
|13|Rule/knowledge-based symptom tagging (`src/preprocessing` or a new `src/symptoms/` module) — maps disease labels to a symptom list.|
|14|Grad-CAM wired to the plant + disease classifiers (`src/explainability/` or inside `classification/`) — first heatmap outputs saved to `reports/`. This is an early, visually strong milestone worth demoing.|
|15|Symptom + explainability outputs combined into one structured JSON result per image (matches `API\_PLAN.md`'s intended output shape).|
|16|Integration test: raw image in → plant + disease + symptoms + heatmap out, run via a single script (`scripts/run\_pipeline.py`).|

## Month 5 — Pest Detection (Phase 4)

|Week|Deliverable|
|-|-|
|17|Bbox-labeled pest dataset confirmed (e.g. IP102 subset) — update `DATASET\_PLAN.md`.|
|18|YOLO model configured and a first training run completed; checkpoint in `models/pest\_detector/`.|
|19|Evaluation (mAP, precision/recall) in `reports/`; `pest\_detector` status flipped from PLANNED to IMPLEMENTED in `README.md` / `ARCHITECTURE.md` if metrics are real.|
|20|Pest output wired into the same pipeline script from Week 16.|

## Month 6 — Severity + Health Score (Phases 5–6)

|Week|Deliverable|
|-|-|
|21|`src/severity/` — affected-area estimation (heuristic on segmentation/detection masks) + severity category logic.|
|22|`src/scoring/` — health score engine reading `config/health\_score.yaml`, replacing the current placeholder weights with justified ones.|
|23|Health score validated on a batch of sample images with a written sanity-check report (does 90+ look healthy, does 30- look diseased).|
|24|`health\_score` and `pest\_detector` sections of `config.yaml` updated to IMPLEMENTED; buffer week for anything slipped from Months 1–6.|

## Month 7 — Knowledge Base (Phase 7)

|Week|Deliverable|
|-|-|
|25|Agricultural reference sources collected into `data/knowledge/` (treatment guides, prevention protocols) — sources documented, not scraped blindly.|
|26|Chunking + embedding pipeline (`src/rag/` groundwork) — vector store populated (FAISS or equivalent).|
|27|Retrieval quality check — manual test set of questions with expected relevant chunks; results in `reports/`.|
|28|`src/database/` — SQLite schema implemented (`plants`, `analyses`, `predictions`, `symptoms`, `recommendations`) per `ARCHITECTURE.md`.|

## Month 8 — RAG Assistant + Recommendations (Phases 9–10, part 1)

|Week|Deliverable|
|-|-|
|29|`rag.answer()` implemented per the `API\_PLAN.md` contract, LLM choice confirmed and documented.|
|30|RAG answers grounded and cited against `data/knowledge/` — tested against the Week 27 question set.|
|31|`src/recommendations/` — turns findings + RAG context into a structured recommendation list, grounded only in retrieved knowledge.|
|32|End-to-end pipeline script now produces a full report object: identification → disease → pest → symptoms → severity → score → recommendations.|

## Month 9 — Streamlit Application (Phase 10, part 2)

|Week|Deliverable|
|-|-|
|33|`app.py` skeleton — image upload, calls the pipeline script from Week 32, shows raw JSON output.|
|34|UI polish: plant/disease/pest cards, severity badge, health score gauge — mirrors the demo mockup slide from the presentation.|
|35|Grad-CAM heatmap and symptom list rendered in the UI; recommendations panel wired in.|
|36|`database.save\_analysis()` / `get\_history()` wired into the app — a plant's history view.|

## Month 10 — Testing, Hardening, and History (Phases 8/11 wrap-up)

|Week|Deliverable|
|-|-|
|37|Unit + integration test suite in `tests/` covering every `src/` module's public contract from `API\_PLAN.md`.|
|38|Error handling pass — corrupted image, unsupported format, low-confidence results all fail gracefully instead of crashing or guessing.|
|39|Performance pass — inference time measured and recorded (CPU), any obvious bottlenecks addressed.|
|40|Buffer / catch-up week for Months 7–10.|

## Month 11 — Evaluation, Documentation, Reports

|Week|Deliverable|
|-|-|
|41|Full system evaluation on a held-out test set — `reports/evaluation\_summary.md` with only real, run-produced numbers.|
|42|`docs/LIMITATIONS.md` reviewed and updated against what was actually built (not just planned).|
|43|User-facing documentation — how to run the app, how to read a report, what it can't tell you.|
|44|Thesis/graduation report writing — architecture chapter, methodology chapter, results chapter drafted from real artifacts in `reports/`.|

## Month 12 — Polish, Demo, and Delivery

|Week|Deliverable|
|-|-|
|45|UI visual polish pass — matches the presentation's visual identity (dark green, glassmorphism cards, glow accents) as closely as a Streamlit app reasonably can.|
|46|Recorded demo video / live-demo script prepared using the real app, not mockups.|
|47|Final presentation updated with real metrics, real screenshots, and real architecture diagrams instead of placeholders.|
|48|Mock defense / dry run with feedback incorporated.|
|49–52|Buffer for supervisor feedback, last fixes, printing/submission logistics, and any environmental/future-extension teaser if time allows.|

\---

## \## How to track this in the repo

## \- Keep this file (`docs/ROADMAP.md`) as the single source of truth for "what week are we on."

## \- At the end of each week, add a one-line dated entry under a `## Log` section below (create it once you start) — what actually shipped vs. what was planned. This becomes useful evidence for the graduation report later.

## \- Update `README.md`'s Feature Status table and `ARCHITECTURE.md`'s Module Status table whenever a module flips from PLANNED to IMPLEMENTED — don't let those two files drift from reality.

## 

## \---

## 

## \## Log

## 

## Chronological record of what actually shipped vs. what was planned. All

## entries reflect real, run-produced results — no entry is written in advance.

## 

## \### Week 1 — Dataset Foundation (15/09/2026)

## \- Dataset source: \*\*PlantVillage (color)\*\* — 38 classes, 14 crops.

## \- Raw files placed under `data/raw/`.

## \- `scripts/check\_dataset.py` run → output: `reports/dataset\_inspection\_summary.md`

## &#x20; and `reports/dataset\_inspection\_report.json`.

## \- `scripts/remove\_duplicates.py` and `scripts/split\_dataset.py` run.

## \- Split: `data/train/`, `data/validation/`, `data/test/` — 38 classes each.

## &#x20; Test set: 8,176 images.

## 

## \### Week 5 — Plant Classifier Architecture (15/09/2026)

## \- `src/classification/plant\_classifier.py` implemented.

## \- Architecture: \*\*MobileNetV2\*\* with a custom classifier head

## &#x20; (Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→38)).

## \- Backbone-freeze option implemented for fast head-only training.

## 

## \### Week 6 — First Training Run (15/09/2026)

## \- `training/train\_plant\_classifier.py` implemented.

## \- Setup: backbone frozen, Adam (lr=1e-3), 15 epochs, batch size 16,

## &#x20; AMP on GPU, WeightedRandomSampler for class imbalance.

## \- Best checkpoint: `models/plant\_classifier/best\_model.pt`

## &#x20; (epoch 14, val\_acc=\*\*0.9433\*\*).

## \- Class map: `models/plant\_classifier/class\_to\_idx.json`.

## \- Training log: `reports/plant\_classifier\_training\_log.csv`.

## \- Training report: `reports/plant\_classifier\_training\_report.md`.

## 

## \### Week 7 — Evaluation (15/09/2026)

## \- `training/evaluate.py` implemented.

## \- Evaluated on the held-out test set (8,176 images).

## \- \*\*Results:\*\*

## &#x20; - Accuracy: \*\*0.9410\*\*

## &#x20; - Top-5 accuracy: \*\*0.9974\*\*

## &#x20; - F1 (macro): \*\*0.9270\*\*

## &#x20; - F1 (weighted): \*\*0.9405\*\*

## \- Outputs:

## &#x20; - `reports/plant\_classifier\_evaluation.md`

## &#x20; - `reports/plant\_classifier\_metrics.json`

## &#x20; - `reports/plant\_classifier\_classification\_report.txt`

## &#x20; - `reports/plant\_classifier\_confusion\_matrix.png`

## 

## \### Week 8 — Augmentation Experiment (15/09/2026) — NEGATIVE RESULT

## \- Tested stronger augmentation (RandomPerspective, GaussianBlur,

## &#x20; RandomErasing, stronger ColorJitter) to improve real-world robustness.

## \- Ran `train\_plant\_classifier.py --epochs 20`.

## \- Result: best val\_acc = \*\*0.8865\*\*, test\_acc = \*\*0.8874\*\*.

## \- \*\*Conclusion:\*\* Augmentation alone is not sufficient for real-world photos.

## &#x20; Reverted to the V1 checkpoint (94.1%). A real-world dataset (either

## &#x20; PlantDoc or user-collected field photos) is required for meaningful

## &#x20; improvement. Recorded here as a real, negative experiment result.

## 

## \### Week 9 — Streamlit Preview App (15/09/2026)

## \- `app.py` implemented with a Green Field UI theme.

## \- Features:

## &#x20; - Image upload + preview

## &#x20; - \*\*Auto-crop\*\* preprocessing (HSV color mask + GrabCut)

## &#x20; - Prediction with plant + condition parsing

## &#x20; - Confidence gauge (SVG)

## &#x20; - Top-5 candidates with ranking

## &#x20; - Per-disease information (severity, description, care advice) for all

## &#x20;   38 classes

## &#x20; - Model info tab with real metrics, worst-5 classes, and confusion matrix

## \- A "Preview build" banner is shown to make the app's scope explicit.

## 

## \### Outstanding / Known Limitations

## \- Real-world field photos (busy backgrounds, whole branches) reduce

## &#x20; accuracy. The 94.1% test accuracy is on PlantVillage-style images only.

## \- Only 38 classes across 14 crops are supported.

## \- No pest detection, severity, health score, or RAG recommendations yet —

## &#x20; those remain PLANNED.

