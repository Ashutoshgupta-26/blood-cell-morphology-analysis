import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import load_image
from src.counting import load_classification_pipeline, analyze_blood_smear
from config import locate_dataset_root

def debug_oversegmentation(img_name):
    root = locate_dataset_root()
    img_path = root / "images/test" / img_name
    
    out_dir = Path("results/debug/problem_image")
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "small_objects"
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    scaler, model, features = load_classification_pipeline()
    res = analyze_blood_smear(img_path, scaler, model, features)
    df = res["predictions_df"]
    
    # Save all instances
    df.to_csv("results/debug/problem_image_instances.csv", index=False)
    
    print(f"Total instances detected: {len(df)}")
    counts = df["predicted_class"].value_counts()
    for k,v in counts.items():
        print(f"{k}: {v}")
        
    # Sort by area and get smallest 20
    smallest = df.sort_values("area").head(20)
    print("\nSmallest 20 objects:")
    img_bgr = load_image(img_path)
    
    for _, row in smallest.iterrows():
        i_id = row["instance_id"]
        cls = row["predicted_class"]
        area = row["area"]
        diam = row["equivalent_diameter"]
        circ = row["circularity"]
        sol = row["solidity"]
        
        print(f"ID {i_id}: {cls} | Area: {area} | Diam: {diam:.1f} | Circ: {circ:.2f} | Sol: {sol:.2f}")
        
        # Crop and save
        x, y = int(row["bbox_x"]), int(row["bbox_y"])
        w, h = int(row["bbox_width"]), int(row["bbox_height"])
        
        # Add padding to crop
        pad = 5
        y1 = max(0, y - pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        
        crop = img_bgr[y1:y2, x1:x2]
        cv2.imwrite(str(crop_dir / f"id_{i_id}_{cls}_area_{area}.png"), crop)
        
    # Visualization: ID, class, area
    vis_img = img_bgr.copy()
    for _, row in df.iterrows():
        x, y = int(row["bbox_x"]), int(row["bbox_y"])
        i_id = row["instance_id"]
        cls = row["predicted_class"]
        area = int(row["area"])
        
        text = f"#{i_id} {cls} ({area})"
        cv2.putText(vis_img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
    cv2.imwrite(str(out_dir / "all_instances_with_area.png"), vis_img)
    
if __name__ == "__main__":
    debug_oversegmentation("7f96e833a77559fb8fe76152b2b76660.png")
