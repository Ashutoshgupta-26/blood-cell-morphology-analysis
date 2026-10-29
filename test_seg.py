import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from src.preprocessing import load_image
from src.segmentation import segment_cells
from config import locate_dataset_root

def test_segmentation_methods(img_name):
    root = locate_dataset_root()
    img_path = root / "images/test" / img_name
    lbl_path = root / "labels/test" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
    
    img = load_image(img_path)
    h_img, w_img = img.shape[:2]
    
    out_dir = Path("results/debug/problem_image/segmentation_methods")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ["otsu", "adaptive", "color_range", "kmeans"]
    
    # Load GT boxes
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
                
    channels = ["green", "clahe_green", "grayscale", "hsv_s", "lab_l", "lab_a"]
                
    for ch in channels:
        try:
            seg = segment_cells(img, method="otsu", channel=ch)
            mask = seg["raw_mask"]
            cv2.imwrite(str(out_dir / f"otsu_ch_{ch}.png"), mask)
            
            # Check coverage
            ratios = []
            for x1, y1, x2, y2 in boxes:
                crop = mask[y1:y2, x1:x2]
                if crop.size > 0:
                    ratios.append(np.count_nonzero(crop == 255) / crop.size)
                    
            print(f"Otsu + {ch}")
            if ratios:
                print(f"  Mean GT Box Coverage: {np.mean(ratios):.4f}")
                print(f"  Boxes < 20% coverage: {sum(1 for r in ratios if r < 0.2)}")
                print(f"  Overall Foreground %: {np.count_nonzero(mask == 255) / mask.size:.4f}")
        except Exception as e:
            print(f"Channel {ch} failed: {e}")

if __name__ == "__main__":
    test_segmentation_methods("7f96e833a77559fb8fe76152b2b76660.png")
