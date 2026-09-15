# PlantVision AI — Model Plan

Status: PLANNED — no model has been trained yet. This document will be filled
in with real architecture choices and justifications once Phase 1 (dataset)
is unblocked and Phase 2 (baseline model) begins.

## Candidate architectures (transfer learning, CPU-feasible)

| Task | Candidate | Rationale (to confirm once data is known) |
|---|---|---|
| Plant identification | MobileNetV2 or EfficientNet-B0 | Small, fast on CPU, strong ImageNet transfer baseline |
| Disease classification | Same backbone family as above | Consistent pipeline, easier to compare fairly |
| Pest detection | YOLO (version TBD) | Only if a bbox-labeled dataset is confirmed (Phase 4) |
| Severity | Heuristic first; segmentation model later if data supports it | Avoids fabricating precision the data can't support |

## Selection criteria (per project rules)

- Accuracy / F1 on a held-out test split (never estimated in advance)
- Inference speed on CPU
- Model size (deployment via Streamlit)
- Actual hardware available (no GPU currently)

## What this document will NOT contain until real training happens

Any specific accuracy, F1, precision/recall, or confusion-matrix numbers.
Those belong in `reports/` and are generated, not estimated.
