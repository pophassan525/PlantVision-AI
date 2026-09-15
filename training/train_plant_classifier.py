"""
PlantVision AI — Train the Plant Identification / Disease Classifier

Phase 2 (Week 5-6, updated Week 8). Trains src/classification/plant_classifier.py
on data/train and data/validation (produced by scripts/split_dataset.py).

Week 8 update: stronger data augmentation to improve real-world robustness
(field photos, varied lighting, angles, backgrounds). Expect slightly lower
PlantVillage test accuracy in exchange for better real-image generalization.

Usage:
    python training/train_plant_classifier.py --data_dir data --epochs 20

Outputs:
    models/plant_classifier/best_model.pt
    models/plant_classifier/last_model.pt
    models/plant_classifier/class_to_idx.json
    reports/plant_classifier_training_log.csv
    reports/plant_classifier_training_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.classification.plant_classifier import build_model  # noqa: E402


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PlantVision plant classifier")
    p.add_argument("--data_dir", type=str, default="data",
                    help="Directory containing train/ and validation/ subfolders")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze_backbone", action="store_true", default=True,
                    help="Train only the classification head (fast baseline)")
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false",
                    help="Fine-tune the full backbone instead of freezing it")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="models/plant_classifier")
    p.add_argument("--reports_dir", type=str, default="reports")
    p.add_argument("--no_amp", action="store_true",
                    help="Disable mixed-precision training")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {val_dir} to exist. "
            f"Run scripts/split_dataset.py first."
        )

    # --- Week 8: stronger augmentation for real-world robustness ---
    train_transform = transforms.Compose([
        # Scale / crop: allow leaves at many sizes and aspect ratios
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),

        # Lighting: simulate sun, shade, overcast, warm/cool casts
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05
        ),

        # Camera angle: simulate a phone held at a slight tilt
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),

        # Focus / motion: simulate handshake or slight out-of-focus
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),

        transforms.ToTensor(),

        # Occlusion: simulate another leaf or shadow covering part of the leaf
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),

        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            "Train and validation folders have different class sets — "
            "check scripts/split_dataset.py output."
        )

    # Weighted sampling to counter class imbalance
    class_counts = Counter(label for _, label in train_ds.samples)
    num_classes = len(train_ds.classes)
    class_weights = {c: 1.0 / class_counts[c] for c in range(num_classes)}
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected — training will be much slower on CPU.")

    train_loader, val_loader, classes = build_dataloaders(
        data_dir, args.batch_size, args.num_workers
    )
    num_classes = len(classes)
    print(f"Found {num_classes} classes, "
          f"{len(train_loader.dataset)} train images, "
          f"{len(val_loader.dataset)} validation images")

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    with open(output_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    model = build_model(num_classes=num_classes, pretrained=True,
                         freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    best_val_acc = 0.0
    log_rows = []
    log_path = reports_dir / "plant_classifier_training_log.csv"

    print(f"\nStarting training for {args.epochs} epochs "
          f"(backbone {'frozen' if args.freeze_backbone else 'fine-tuning'}, "
          f"batch_size={args.batch_size}, AMP={'on' if scaler else 'off'})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": round(elapsed, 1),
        })

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "class_to_idx": class_to_idx,
        }, output_dir / "last_model.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
            }, output_dir / "best_model.pt")
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    report_path = reports_dir / "plant_classifier_training_report.md"
    with open(report_path, "w") as f:
        f.write("# Plant Classifier — Training Report\n\n")
        f.write(f"- Epochs run: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Backbone: {'frozen (head only)' if args.freeze_backbone else 'fine-tuned'}\n")
        f.write(f"- Augmentation: STRONG (Week 8 — real-world oriented)\n")
        f.write(f"- Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"- Classes: {num_classes}\n")
        f.write(f"- Full per-epoch log: `{log_path.as_posix()}`\n")
        f.write(f"- Best checkpoint: `{(output_dir / 'best_model.pt').as_posix()}`\n\n")
        f.write("This report is generated directly from the run above — "
                "no numbers here are estimated or invented.\n")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()