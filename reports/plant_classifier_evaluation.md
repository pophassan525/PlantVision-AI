# Plant Classifier — Evaluation Report

- Test set: `data/test`
- Test images: 8176
- Classes: 38
- Checkpoint: `models/plant_classifier/best_model.pt` (epoch 14, val_acc 0.9433)

## Overall metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9410 |
| Top-5 accuracy | 0.9974 |
| Precision (macro) | 0.9215 |
| Recall (macro) | 0.9376 |
| F1 (macro) | 0.9270 |
| F1 (weighted) | 0.9405 |

## Worst 5 classes (by F1)

| Class | F1 |
|---|---|
| Potato___healthy | 0.7419 |
| Tomato___Early_blight | 0.7500 |
| Tomato___Target_Spot | 0.7954 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.8046 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.8395 |

## Artifacts

- `reports/plant_classifier_metrics.json` — all metrics as JSON
- `reports/plant_classifier_classification_report.txt` — per-class sklearn report
- `reports/plant_classifier_confusion_matrix.png` — confusion matrix heatmap

All numbers above were produced by running this script on the held-out test set — none are estimated or hand-written.
