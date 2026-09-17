"""
Quick manual test for UnknownPlantDetector.
Usage:
    python -m src.recognition.test_recognition "path\\to\\some\\image.jpg"
"""

import sys
from .unknown_plant_detector import UnknownPlantDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.recognition.test_recognition <image_path>")
        return

    image_path = sys.argv[1]

    detector = UnknownPlantDetector(
        embeddings_dir="data/embeddings",
        similarity_threshold=0.75,  # tweak this after seeing real scores below
    )

    result = detector.detect(image_path)

    print("\n--- Result ---")
    print(f"Status:     {result.get('status')}")
    print(f"Class:      {result.get('class')}")
    print(f"Confidence: {result.get('confidence')}")

    all_scores = result.get("all_scores", {})
    if all_scores:
        print("\nTop 5 matches:")
        top5 = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for class_name, score in top5:
            print(f"  {class_name}: {score:.4f}")


if __name__ == "__main__":
    main()
