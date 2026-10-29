import os
import pandas as pd
import numpy as np
from src.preprocessing import load_image
from src.segmentation import segment_cells
from config import locate_dataset_root

def evaluate_morphology():
    root = locate_dataset_root()
    val_images = os.listdir(root / "images/val")
    
    results = []
    
    for img_name in val_images:
        img_path = root / "images/val" / img_name
        lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        
        img = load_image(img_path)
        h_img, w_img = img.shape[:2]
        
        # Load GT
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = max(0, int((cx - w/2) * w_img))
                    y1 = max(0, int((cy - h/2) * h_img))
                    x2 = min(w_img, int((cx + w/2) * w_img))
                    y2 = min(h_img, int((cy + h/2) * h_img))
                    boxes.append((x1, y1, x2, y2))
                    
        # Test original
        seg_orig = segment_cells(img, options={"close_before_fill": False})
        mask_orig = seg_orig["cleaned_mask"]
        orig_ratios = [np.count_nonzero(mask_orig[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1, y1, x2, y2 in boxes if x2>x1 and y2>y1]
        
        # Test close_before_fill
        seg_new = segment_cells(img, options={"close_before_fill": True})
        mask_new = seg_new["cleaned_mask"]
        new_ratios = [np.count_nonzero(mask_new[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1, y1, x2, y2 in boxes if x2>x1 and y2>y1]
        
        results.append({
            "image": img_name,
            "orig_mean_cov": np.mean(orig_ratios) if orig_ratios else 0,
            "new_mean_cov": np.mean(new_ratios) if new_ratios else 0,
            "orig_bad_boxes": sum(1 for r in orig_ratios if r < 0.2),
            "new_bad_boxes": sum(1 for r in new_ratios if r < 0.2),
        })
        
    df = pd.DataFrame(results)
    print("Mean Coverage Original:", df["orig_mean_cov"].mean())
    print("Mean Coverage New:", df["new_mean_cov"].mean())
    print("Total Bad Boxes Original:", df["orig_bad_boxes"].sum())
    print("Total Bad Boxes New:", df["new_bad_boxes"].sum())
    
if __name__ == "__main__":
    evaluate_morphology()
