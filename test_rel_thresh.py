import cv2, numpy as np, os, pandas as pd
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter
from src.segmentation import otsu_segmentation
from src.morphology import fill_holes
from config import locate_dataset_root

def test_relative_threshold():
    root = locate_dataset_root()
    val_images = os.listdir(root / "images/val")[:15]
    
    results = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for img_name in val_images:
        img_path = root / "images/val" / img_name
        lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        if not lbl_path.exists(): continue
        
        img = load_image(img_path)
        g = img[:,:,1]
        
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = max(0, int((cx - w/2) * img.shape[1]))
                    y1 = max(0, int((cy - h/2) * img.shape[0]))
                    x2 = min(img.shape[1], int((cx + w/2) * img.shape[1]))
                    y2 = min(img.shape[0], int((cy + h/2) * img.shape[0]))
                    boxes.append((x1, y1, x2, y2))
                    
        t_otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        for mult in [1.0, 1.05, 1.10, 1.15, 1.20]:
            t_adj = min(254, t_otsu * mult)
            _, m = cv2.threshold(g, t_adj, 255, cv2.THRESH_BINARY_INV)
            
            # Removed normalize_polarity to prevent false inversions on high thresholds
                
            c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            f = fill_holes(c)
            
            num, _ = cv2.connectedComponents(f)
            r = [np.count_nonzero(f[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
            cov = np.mean(r) if r else 0
            
            results.append({
                "Image": img_name,
                "Multiplier": mult,
                "Cov": cov,
                "Comp": num - 1,
                "GT": len(boxes)
            })
            
    df = pd.DataFrame(results)
    agg = df.groupby("Multiplier").agg({"Cov": "mean", "Comp": "mean", "GT": "mean"})
    print(agg)

if __name__ == "__main__":
    test_relative_threshold()
