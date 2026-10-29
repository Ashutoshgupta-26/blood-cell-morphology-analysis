import cv2
import numpy as np
import os
import pandas as pd
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter
from src.segmentation import otsu_segmentation
from src.morphology import fill_holes
from config import locate_dataset_root

def test_closing_sizes():
    root = locate_dataset_root()
    val_images = ["7f96e833a77559fb8fe76152b2b76660.png"]
    root_dir = root / "images/test"
    lbl_dir = root / "labels/test"
    
    kernels = [15, 21, 31, 41]
    
    results = []
    
    for img_name in val_images:
        img_path = root_dir / img_name
        lbl_path = lbl_dir / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        if not lbl_path.exists(): continue
        
        img = load_image(img_path)
        h_img, w_img = img.shape[:2]
        
        from src.preprocessing import clahe_enhancement
        green = grayscale_conversion(img, 'green')
        filtered = bilateral_filter(green)
        pre = clahe_enhancement(filtered)
        _, raw = otsu_segmentation(pre)
        
        # Load GT boxes
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
                    
        for k in kernels:
            if k > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
            else:
                closed = raw.copy()
                
            filled = fill_holes(closed) # Using our new CCOMP fill_holes
            
            # Coverage
            covs = [np.count_nonzero(filled[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
            mean_cov = np.mean(covs) if covs else 0
            bad_boxes = sum(1 for c in covs if c < 0.2)
            
            # Number of components
            num_comp, _ = cv2.connectedComponents(filled)
            
            results.append({
                "image": img_name,
                "kernel": k,
                "mean_cov": mean_cov,
                "bad_boxes": bad_boxes,
                "components": num_comp - 1,
                "gt_count": len(boxes)
            })
            
    df = pd.DataFrame(results)
    
    print("Aggregate Results by Kernel Size:")
    agg = df.groupby("kernel").agg({
        "mean_cov": "mean",
        "bad_boxes": "sum",
        "components": "mean",
        "gt_count": "mean"
    })
    print(agg)

if __name__ == "__main__":
    test_closing_sizes()
