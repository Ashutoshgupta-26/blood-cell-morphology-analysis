"""
Dataset Utility Functions for TXL-PBC Blood Smear Dataset.
Provides directory discovery, image-label pairing verification, missing file detection,
and statistical analysis across train, validation, and test splits.
"""

import sys
from pathlib import Path
from collections import Counter
import pandas as pd

# Add parent directory to path so config can be imported if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATASET_ROOT, load_dataset_classes


def get_dataset_root(custom_root=None):
    """
    Get resolved dataset root Path object.
    
    Args:
        custom_root (str or Path, optional): Custom path to TXL-PBC directory.
        
    Returns:
        Path: Resolved dataset root path.
    """
    if custom_root:
        root = Path(custom_root)
        if root.exists():
            return root
    if DATASET_ROOT and DATASET_ROOT.exists():
        return DATASET_ROOT
    raise FileNotFoundError("TXL-PBC dataset root directory not found.")


def list_split_files(dataset_root=None, split="train"):
    """
    List all image files and corresponding annotation label files in a given split.
    
    Args:
        dataset_root (Path, optional): Path to TXL-PBC dataset root.
        split (str): Split name ('train', 'val', or 'test').
        
    Returns:
        tuple: (list of image Path objects, list of label Path objects)
    """
    root = get_dataset_root(dataset_root)
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images split directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels split directory does not exist: {labels_dir}")

    # Support png, jpg, jpeg extensions with case-insensitive deduplication
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    images_set = set()
    for ext in image_extensions:
        images_set.update(p.resolve() for p in images_dir.glob(ext))
    images = sorted(list(images_set))

    labels = sorted(list(labels_dir.glob("*.txt")))

    return images, labels


def get_paired_files(dataset_root=None, split="train"):
    """
    Match each image with its corresponding annotation file by stem name.
    
    Args:
        dataset_root (Path, optional): Path to TXL-PBC dataset root.
        split (str): Split name ('train', 'val', or 'test').
        
    Returns:
        list of dict: List of pairs with 'stem', 'image_path', 'label_path', and 'has_label' status.
    """
    images, labels = list_split_files(dataset_root, split)
    label_map = {lbl.stem: lbl for lbl in labels}

    pairs = []
    for img_path in images:
        stem = img_path.stem
        lbl_path = label_map.get(stem)
        pairs.append({
            "stem": stem,
            "image_path": img_path,
            "label_path": lbl_path,
            "has_label": lbl_path is not None and lbl_path.exists()
        })
    return pairs


def verify_dataset_pairing(dataset_root=None):
    """
    Verify pairing between images and label files across train, val, and test splits.
    
    Returns:
        dict: Detailed report per split showing total images, labels, matched, missing labels, missing images.
    """
    root = get_dataset_root(dataset_root)
    splits = ["train", "val", "test"]
    report = {}

    for split in splits:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split

        image_files = set(f.stem for ext in ["*.png", "*.jpg", "*.jpeg"] for f in images_dir.glob(ext))
        label_files = set(f.stem for f in labels_dir.glob("*.txt"))

        matched = image_files.intersection(label_files)
        missing_labels = image_files - label_files
        orphan_labels = label_files - image_files

        report[split] = {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "matched_pairs": len(matched),
            "missing_labels_count": len(missing_labels),
            "orphan_labels_count": len(orphan_labels),
            "missing_labels": sorted(list(missing_labels)),
            "orphan_labels": sorted(list(orphan_labels)),
        }

    return report


def count_class_instances_in_split(dataset_root=None, split="train"):
    """
    Count cell instances by class ID across all annotation files in a split.
    
    Args:
        dataset_root (Path, optional): Path to TXL-PBC dataset root.
        split (str): Split name ('train', 'val', or 'test').
        
    Returns:
        Counter: Counter mapping class_id (int) -> frequency count.
    """
    _, labels = list_split_files(dataset_root, split)
    class_counts = Counter()

    for label_path in labels:
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        except Exception as e:
            print(f"Warning: Failed reading label file {label_path}: {e}")

    return class_counts


def get_full_dataset_statistics(dataset_root=None):
    """
    Compute comprehensive dataset statistics for all splits.
    
    Returns:
        dict: Summary statistics including split counts, instance counts per class, and class names.
    """
    root = get_dataset_root(dataset_root)
    class_names = load_dataset_classes(root / "data.yaml")
    splits = ["train", "val", "test"]

    pairing_report = verify_dataset_pairing(root)
    stats = {
        "dataset_root": str(root),
        "class_mapping": class_names,
        "splits": {},
        "overall_instances": Counter(),
    }

    for split in splits:
        class_counts = count_class_instances_in_split(root, split)
        stats["overall_instances"].update(class_counts)

        # Convert class_id keys to class_name for clarity
        named_counts = {class_names.get(cid, f"Class_{cid}"): count for cid, count in class_counts.items()}
        
        stats["splits"][split] = {
            "num_images": pairing_report[split]["total_images"],
            "num_labels": pairing_report[split]["total_labels"],
            "matched_pairs": pairing_report[split]["matched_pairs"],
            "missing_labels": pairing_report[split]["missing_labels_count"],
            "orphan_labels": pairing_report[split]["orphan_labels_count"],
            "class_distribution": named_counts,
            "total_instances": sum(class_counts.values()),
        }

    return stats


def print_dataset_statistics(stats=None, dataset_root=None):
    """
    Print formatted dataset statistics to console.
    """
    if stats is None:
        stats = get_full_dataset_statistics(dataset_root)

    print("\n" + "=" * 65)
    print(" TXL-PBC DATASET STATISTICAL ANALYSIS REPORT ")
    print("=" * 65)
    print(f"Dataset Root: {stats['dataset_root']}")
    print(f"Class Definitions: {stats['class_mapping']}")
    print("-" * 65)

    # Build Pandas DataFrame for visual table formatting
    rows = []
    for split, info in stats["splits"].items():
        row = {
            "Split": split.capitalize(),
            "Images": info["num_images"],
            "Labels": info["num_labels"],
            "Matched": info["matched_pairs"],
            "Total Cells": info["total_instances"],
        }
        for cid, name in stats["class_mapping"].items():
            row[f"Cells ({name})"] = info["class_distribution"].get(name, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("-" * 65)

    # Verification checks
    all_matched = all(info["missing_labels"] == 0 and info["orphan_labels"] == 0 for info in stats["splits"].values())
    if all_matched:
        print("[OK] Pairing Check: ALL images and label files are 100% correctly paired across all splits!")
    else:
        print("! Warning: Found missing or un-paired files in dataset splits.")
        for split, info in stats["splits"].items():
            if info["missing_labels"] > 0:
                print(f"  - {split}: {info['missing_labels']} images missing label files")
            if info["orphan_labels"] > 0:
                print(f"  - {split}: {info['orphan_labels']} label files missing images")
    print("=" * 65 + "\n")
