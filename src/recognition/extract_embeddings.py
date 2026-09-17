"""
Script to extract and save embeddings for all known plant classes.
Averages ALL images inside each class folder for a more robust, realistic embedding.
Run this once (or whenever you add/replace training images) to pre-compute embeddings.
"""

import os
import numpy as np
from pathlib import Path
from .clip_embedder import CLIPEmbedder

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def extract_and_save_embeddings(data_dir="data/train", embeddings_dir="data/embeddings"):
    """Extract averaged embeddings from training data."""

    data_path = Path(data_dir).resolve()
    embeddings_path = Path(embeddings_dir).resolve()

    print(f"Looking for training data in: {data_path}")

    if not data_path.exists():
        print(f"✗ ERROR: data folder does not exist: {data_path}")
        print("  Make sure you are running this command from the project root")
        print("  (the folder that contains 'data\\train'), e.g.:")
        print("    python -m src.recognition.extract_embeddings")
        return

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"✗ ERROR: no class subfolders found inside {data_path}")
        print("  Expected structure: data/train/<plant_name>/image1.jpg ...")
        return

    print(f"Found {len(class_dirs)} class folders. Loading CLIP model...")
    clip = CLIPEmbedder()
    os.makedirs(embeddings_path, exist_ok=True)

    saved_count = 0

    for class_dir in class_dirs:
        class_name = class_dir.name

        image_files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )

        print(f"Processing '{class_name}' ({len(image_files)} images)...", end=" ")

        if not image_files:
            print("✗ (no images found)")
            continue

        embeddings = []
        for image_file in image_files:
            embedding = clip.get_image_embedding(str(image_file))
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            print("✗ (all images failed to process)")
            continue

        # Average across all successfully processed images, then re-normalize
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        np.save(embeddings_path / f"{class_name}.npy", avg_embedding)
        saved_count += 1
        print(f"✓ (averaged {len(embeddings)} images)")

    print(f"\nDone. Saved {saved_count}/{len(class_dirs)} class embeddings to {embeddings_path}")


if __name__ == "__main__":
    extract_and_save_embeddings()
