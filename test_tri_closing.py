"""Test closing kernel sizes on Triangle mask across val images."""
import cv2, numpy as np, os, pandas as pd
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter
from src.morphology import fill_holes
from config import locate_dataset_root

root = locate_dataset_root()
val_images = sorted(os.listdir(root / "images/val"))[:15]

rows = []
for ks in [5, 7, 9, 11]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    covs_all = []
    comps_all = []
    for img_name in val_images:
        img_path = root / "images/val" / img_name
        lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        if not lbl_path.exists(): continue
        img = load_image(img_path)
        g = bilateral_filter(grayscale_conversion(img, 'green'))
        h_img, w_img = img.shape[:2]
        
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = max(0, int((cx - w/2) * w_img))
                    y1 = max(0, int((cy - h/2) * h_img))
                    x2 = min(w_img, int((cx + w/2) * w_img))
                    y2 = min(h_img, int((cy + h/2) * h_img))
                    boxes.append((x1, y1, x2, y2))
        
        _, tri_mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        closed = cv2.morphologyEx(tri_mask, cv2.MORPH_CLOSE, k)
        filled = fill_holes(closed)
        
        covs = [np.count_nonzero(filled[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1))
                for x1,y1,x2,y2 in boxes if (x2-x1)*(y2-y1) > 0]
        num, _ = cv2.connectedComponents(filled)
        covs_all.append(np.mean(covs) if covs else 0)
        comps_all.append(num - 1)
    
    rows.append({
        "Kernel": ks,
        "Mean_Cov": round(np.mean(covs_all), 3),
        "Mean_Comp": round(np.mean(comps_all), 1),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
