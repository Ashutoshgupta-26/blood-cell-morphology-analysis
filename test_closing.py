import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from src.preprocessing import load_image, clahe_enhancement, bilateral_filter, grayscale_conversion
from src.segmentation import otsu_segmentation
from src.morphology import fill_holes
from config import locate_dataset_root

def test_large_closing(img_name):
    root = locate_dataset_root()
    img_path = root / "images/test" / img_name
    lbl_path = root / "labels/test" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
    
    img = load_image(img_path)
    h_img, w_img = img.shape[:2]
    
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
                
    pre = clahe_enhancement(bilateral_filter(grayscale_conversion(img, 'green')))
    _, raw = otsu_segmentation(pre)
    
    kernels = [5, 9, 15, 21, 31]
    
    for k in kernels:
        bridge = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        filled = fill_holes(bridge)
        
        ratios = []
        for x1, y1, x2, y2 in boxes:
            crop = filled[y1:y2, x1:x2]
            if crop.size > 0:
                ratios.append(np.count_nonzero(crop == 255) / crop.size)
                
        print(f"Kernel {k}x{k}:")
        if ratios:
            print(f"  Mean GT Box Coverage: {np.mean(ratios):.4f}")
            print(f"  Boxes < 20% coverage: {sum(1 for r in ratios if r < 0.2)}")
            print(f"  Boxes > 60% coverage: {sum(1 for r in ratios if r > 0.6)}")

if __name__ == "__main__":
    test_large_closing("7f96e833a77559fb8fe76152b2b76660.png")
