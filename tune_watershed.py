import pandas as pd
import numpy as np
import cv2
from pathlib import Path

from src.preprocessing import load_image
from src.segmentation import segment_cells
from src.watershed import separate_touching_cells
from config import locate_dataset_root, WATERSHED_CONFIG
import os

def tune_watershed_threshold():
    root = locate_dataset_root()
    val_images = os.listdir(root / "images/val")
    
    ratios = [0.1, 0.25, 0.5, 0.7]
    results = []
    
    for ratio in ratios:
        print(f"Testing ratio: {ratio}...")
        total_gt_cells = 0
        total_pred_instances = 0
        
        for img_name in val_images:
            img_path = root / "images/val" / img_name
            lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
            
            # Count GT
            if lbl_path.exists():
                with open(lbl_path) as f:
                    total_gt_cells += len([line for line in f if line.strip()])
            
            # Predict
            img = load_image(img_path)
            seg = segment_cells(img)
            ws = separate_touching_cells(seg["cleaned_mask"], img, threshold_ratio=ratio)
            
            total_pred_instances += ws["labels"].max()
            
        results.append({
            "ratio": ratio,
            "val_gt_cells": total_gt_cells,
            "val_pred_instances": total_pred_instances,
            "MAE": abs(total_pred_instances - total_gt_cells)
        })
        
    df = pd.DataFrame(results)
    print("\nWatershed Threshold Tuning Results (Validation Set):")
    print(df)
    
if __name__ == "__main__":
    tune_watershed_threshold()
