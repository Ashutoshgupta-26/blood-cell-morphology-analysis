"""V1 vs V2 regression across 15 validation images."""
import cv2, numpy as np, os, pandas as pd
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter
from src.morphology import fill_holes, clean_segmentation_mask
from config import locate_dataset_root

root = locate_dataset_root()
val_images = sorted(os.listdir(root / "images/val"))[:15]
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

rows = []
for img_name in val_images:
    img_path = root / "images/val" / img_name
    lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
    if not lbl_path.exists(): continue
    
    img = load_image(img_path)
    g = grayscale_conversion(img, 'green')
    filtered = bilateral_filter(g)
    
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
    
    def eval_mask(mask):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        filled = fill_holes(closed)
        covs = [np.count_nonzero(filled[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) 
                for x1,y1,x2,y2 in boxes if (x2-x1)*(y2-y1) > 0]
        num, _ = cv2.connectedComponents(filled)
        return np.mean(covs) if covs else 0, num - 1
    
    # V1: Otsu
    _, otsu_mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_cov, otsu_comp = eval_mask(otsu_mask)
    
    # V2: Triangle
    _, tri_mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    tri_cov, tri_comp = eval_mask(tri_mask)
    
    rows.append({
        "Image": img_name[:12],
        "GT": len(boxes),
        "Otsu_Cov": round(otsu_cov, 3),
        "Otsu_Comp": otsu_comp,
        "Tri_Cov": round(tri_cov, 3),
        "Tri_Comp": tri_comp,
        "Cov_Gain": round(tri_cov - otsu_cov, 3)
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nOtsu Mean Coverage: {df['Otsu_Cov'].mean():.3f}")
print(f"Triangle Mean Coverage: {df['Tri_Cov'].mean():.3f}")
print(f"Mean Gain: {df['Cov_Gain'].mean():.3f}")
