"""
PlantVision AI — Plant Classifier Evaluation

Phase 2 (Week 7). Evaluates the trained plant classifier on the held-out
test set (data/test/) — images the model never saw during training or
validation.

Outputs (all written to reports/):
  - plant_classifier_evaluation.md
  - plant_classifier_metrics.json
  - plant_classifier_classification_report.txt
  - plant_classifier_confusion_matrix.png

All numbers are produced by this script only — never hand-written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.classification.plant_classifier import build_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the plant classifier on data/test/")
    p.add_argument("--checkpoint", type=str, default="models/plant_classifier/best_model.pt")
    p.add_argument("--test_dir", type=str, default="data/test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--reports_dir", type=str, default="reports")
    return p.parse_args()


def build_test_loader(test_dir: Path, batch_size: int, num_workers: int):
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(test_dir, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, ds.classes


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes} | trained for {checkpoint['epoch']} epochs | "
          f"checkpoint val_acc: {checkpoint['val_acc']:.4f}")

    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_loader, folder_classes = build_test_loader(
        Path(args.test_dir), args.batch_size, args.num_workers
    )
    print(f"Test images: {len(test_loader.dataset)}")

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(y_true, y_pred)
    top5 = top_k_accuracy_score(y_true, y_probs, k=5, labels=list(range(num_classes)))
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n=== Overall metrics ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Top-5 accuracy : {top5:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro)   : {recall_macro:.4f}")
    print(f"F1 (macro)       : {f1_macro:.4f}")
    print(f"F1 (weighted)    : {f1_weighted:.4f}")

    target_names = [idx_to_class[i] for i in range(num_classes)]
    report_txt = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=target_names, digits=4, zero_division=0,
    )
    print("\n=== Per-class report (first 10 lines) ===")
    print("\n".join(report_txt.splitlines()[:10]))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True,
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Plant Classifier — Confusion Matrix (test set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    cm_path = reports_dir / "plant_classifier_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    f1_per_class = f1_score(y_true, y_pred, average=None,
                            labels=list(range(num_classes)), zero_division=0)
    worst5_idx = np.argsort(f1_per_class)[:5]

    metrics = {
        "accuracy": float(acc),
        "top5_accuracy": float(top5),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "num_classes": num_classes,
        "num_test_images": int(len(y_true)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_acc": float(checkpoint["val_acc"]),
        "per_class_f1": {idx_to_class[i]: float(f1_per_class[i]) for i in range(num_classes)},
        "worst_5_classes": [
            {"class": idx_to_class[int(i)], "f1": float(f1_per_class[int(i)])}
            for i in worst5_idx
        ],
    }
    metrics_path = reports_dir / "plant_classifier_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report_path = reports_dir / "plant_classifier_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    md_path = reports_dir / "plant_classifier_evaluation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Plant Classifier — Evaluation Report\n\n")
        f.write(f"- Test set: `{Path(args.test_dir).as_posix()}`\n")
        f.write(f"- Test images: {len(y_true)}\n")
        f.write(f"- Classes: {num_classes}\n")
        f.write(f"- Checkpoint: `{ckpt_path.as_posix()}` "
                f"(epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.4f})\n\n")
        f.write("## Overall metrics\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Accuracy | {acc:.4f} |\n")
        f.write(f"| Top-5 accuracy | {top5:.4f} |\n")
        f.write(f"| Precision (macro) | {precision_macro:.4f} |\n")
        f.write(f"| Recall (macro) | {recall_macro:.4f} |\n")
        f.write(f"| F1 (macro) | {f1_macro:.4f} |\n")
        f.write(f"| F1 (weighted) | {f1_weighted:.4f} |\n\n")
        f.write("## Worst 5 classes (by F1)\n\n")
        f.write("| Class | F1 |\n|---|---|\n")
        for row in metrics["worst_5_classes"]:
            f.write(f"| {row['class']} | {row['f1']:.4f} |\n")
        f.write("\n## Artifacts\n\n")
        f.write(f"- `{metrics_path.as_posix()}` — all metrics as JSON\n")
        f.write(f"- `{report_path.as_posix()}` — per-class sklearn report\n")
        f.write(f"- `{cm_path.as_posix()}` — confusion matrix heatmap\n\n")
        f.write("All numbers above were produced by running this script on the "
                "held-out test set — none are estimated or hand-written.\n")

    print("\n=== Artifacts written ===")
    print(f"  {md_path}")
    print(f"  {metrics_path}")
    print(f"  {report_path}")
    print(f"  {cm_path}")


if __name__ == "__main__":
    main()