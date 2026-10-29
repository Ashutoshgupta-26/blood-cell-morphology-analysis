import pandas as pd
import numpy as np
import cv2
from pathlib import Path

from src.preprocessing import load_image
from src.segmentation import segment_cells
from config import locate_dataset_root

def check_gt_coverage(img_name):
    root = locate_dataset_root()
    img_path = root / "images/test" / img_name
    lbl_path = root / "labels/test" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
    
    img = load_image(img_path)
    h_img, w_img = img.shape[:2]
    
    # Run segmentation
    seg = segment_cells(img)
    mask = seg["cleaned_mask"]
    
    # Load GT
    gt_stats = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            lines = f.readlines()
        with open(root / "data.yaml") as f:
            import yaml
            class_names = yaml.safe_load(f)["names"]
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts: continue
            
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # Denormalize
            x1 = max(0, int((cx - w/2) * w_img))
            y1 = max(0, int((cy - h/2) * h_img))
            x2 = min(w_img, int((cx + w/2) * w_img))
            y2 = min(h_img, int((cy + h/2) * h_img))
            
            # Crop mask
            crop = mask[y1:y2, x1:x2]
            box_area = crop.size
            if box_area > 0:
                fg_pixels = np.count_nonzero(crop == 255)
                fg_ratio = fg_pixels / box_area
            else:
                fg_ratio = 0
                
            gt_stats.append({
                "box_id": i,
                "class": class_names[cls_id],
                "box_area": box_area,
                "fg_ratio": round(fg_ratio, 4)
            })
            
    df = pd.DataFrame(gt_stats)
    
    out_dir = Path("results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "segmentation_gt_coverage.csv", index=False)
    
    print(df)
    
if __name__ == "__main__":
    check_gt_coverage("7f96e833a77559fb8fe76152b2b76660.png")
