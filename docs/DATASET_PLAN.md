# PlantVision AI — Dataset Plan

Status: **BLOCKED — no dataset source confirmed yet.** This document is intentionally
mostly TODO. Nothing below should be treated as a real dataset until it is filled in
against an actual, inspected source.

## Why this is blocked right now

Phase 1 cannot start honestly until a real image dataset is identified and inspected.
The candidate public datasets from the project brief are:

- **PlantVillage** — common baseline, large, mostly lab/controlled-background images
- **PlantDoc** — smaller, real-world/field images with more visual noise
- **IP102** — pest-focused, useful for a future pest-detection phase

None of these are currently present in this environment, and the code-execution
sandbox this project is being built in does not have general internet access to
arbitrary dataset hosts (e.g. Kaggle). Two realistic paths forward:

1. You upload a dataset (or a representative subset) directly.
2. We use a GitHub-hosted mirror of one of the above (GitHub is reachable from
   this environment) — but this still needs to be located and its license checked
   before use.

## Required fields once a source is chosen

- Dataset source & URL: `TODO`
- License: `TODO`
- Crop classes: `TODO`
- Disease classes: `TODO`
- Number of images (per class): `TODO`
- Image quality notes (resolution, background consistency, lighting): `TODO`
- Label quality notes (verified vs. crowd-labeled): `TODO`
- Train / validation / test split ratios: `TODO`
- Class imbalance analysis: `TODO`
- Augmentation strategy: `TODO`
- Known limitations of this dataset: `TODO`

## Inspection checklist (run before committing to a dataset)

- [ ] Class names match real, distinguishable categories
- [ ] Image counts per class are sufficient (flag any class under ~100 images)
- [ ] No duplicate images across train/val/test (would leak and inflate metrics)
- [ ] No corrupted/unreadable files
- [ ] Labels are plausible on manual spot-check
- [ ] License permits the intended use (academic project)

## Next step

## Dataset Verification Log

- **Date verified:** 2026-09-14
- **Source:** github.com/spMohanty/PlantVillage-Dataset (color subset)
- **Classes found:** 38
- **Total valid images:** 54,284
- **Corrupted files:** 0
- **Duplicate groups (after cleanup):** 0
- **Imbalanced classes:** 13 (range: 275 – 5,507 images per class; largest gap ~20x
  between `Cedar_apple_rust` and `Orange_Huanglongbing` — to be addressed with
  class weighting or weighted sampling during training, not now)
- **Split:** train / validation / test folders created under `data/`
- **Full reports:** `reports/dataset_inspection_report.json`,
  `reports/dataset_inspection_summary.md`
  
  ## Plant Classifier — Training Log (Week 5-6)

- **Date:** 2026-09-15
- **Architecture:** MobileNetV2 (ImageNet pretrained), backbone frozen, custom head
- **Dataset:** 37,982 train / 8,126 validation images, 38 classes
- **Config:** batch_size=16, lr=1e-3, 15 epochs, AMP on, weighted sampling for class imbalance
- **Result:** Best val_acc = 0.9433 (Epoch 14/15)
- **Observation:** val_acc consistently ≥ train_acc throughout — no overfitting signs.
  Curve plateaued from epoch 12 onward; frozen-backbone baseline likely near its ceiling.
- **Checkpoint:** `models/plant_classifier/best_model.pt`
- **Full log:** `reports/plant_classifier_training_log.csv`,
  `reports/plant_classifier_training_report.md`