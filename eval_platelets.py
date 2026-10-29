import os
import pandas as pd
import numpy as np
import yaml
from src.preprocessing import load_image
from src.segmentation import segment_cells
from config import locate_dataset_root

def eval_platelets():
    root = locate_dataset_root()
    val_images = os.listdir(root / "images/val")
    
    with open(root / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]
        
    platelet_id = class_names.index("Platelets")
    
    results = []
    
    for img_name in val_images:
        img_path = root / "images/val" / img_name
        lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        
        if not lbl_path.exists(): continue
        
        img = load_image(img_path)
        h_img, w_img = img.shape[:2]
        
        # Load Platelet GT
        platelet_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if int(parts[0]) == platelet_id:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = max(0, int((cx - w/2) * w_img))
                    y1 = max(0, int((cy - h/2) * h_img))
                    x2 = min(w_img, int((cx + w/2) * w_img))
                    y2 = min(h_img, int((cy + h/2) * h_img))
                    platelet_boxes.append((x1, y1, x2, y2))
                    
        if not platelet_boxes: continue
        
        seg = segment_cells(img)
        mask = seg["cleaned_mask"]
        
        for x1, y1, x2, y2 in platelet_boxes:
            crop = mask[y1:y2, x1:x2]
            if crop.size > 0:
                cov = np.count_nonzero(crop == 255) / crop.size
                results.append({"image": img_name, "coverage": cov})
                
    df = pd.DataFrame(results)
    if not df.empty:
        print(f"Total Platelets: {len(df)}")
        print(f"Mean Platelet Coverage: {df['coverage'].mean():.4f}")
        print(f"Platelets < 20% coverage (Lost): {sum(df['coverage'] < 0.2)}")
    else:
        print("No platelets found in validation set.")

if __name__ == "__main__":
    eval_platelets()
