"""
Notebook/Exploration Script for Dataset Inspection and Annotation Verification.
Run this script to quickly inspect individual dataset images and print annotation coordinates.
"""

import sys
from pathlib import Path
import cv2

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.dataset_utils import list_split_files, get_paired_files, print_dataset_statistics
from src.annotation_utils import parse_yolo_annotation
from src.visualization import create_side_by_side_comparison


def main():
    print("=== TXL-PBC Dataset Exploration ===")
    print_dataset_statistics()

    train_pairs = get_paired_files(split="train")
    print(f"\nTotal paired train files: {len(train_pairs)}")

    # Inspect first 3 training pairs
    for idx, pair in enumerate(train_pairs[:3], 1):
        img_path = pair["image_path"]
        lbl_path = pair["label_path"]

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        anns = parse_yolo_annotation(lbl_path, w, h)

        print(f"\nSample {idx}: {img_path.name}")
        print(f"  - Dimensions: {w} x {h}")
        print(f"  - Annotations Count: {len(anns)}")
        for ann in anns[:5]:  # print first 5 annotations
            print(f"    - {ann['class_name']} (Class {ann['class_id']}): Pixel Box [{ann['x1']}, {ann['y1']}, {ann['x2']}, {ann['y2']}]")

        out_name = f"exploration_sample_{idx:03d}.png"
        _, _, saved_path = create_side_by_side_comparison(img_path, lbl_path, save_filename=out_name)
        print(f"  - Saved visualization: {saved_path}")


if __name__ == "__main__":
    main()
