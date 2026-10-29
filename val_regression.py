import os
import pandas as pd
from src.counting import load_classification_pipeline, analyze_blood_smear
from config import locate_dataset_root
from pathlib import Path

def validate_pipeline():
    root = locate_dataset_root()
    val_images = os.listdir(root / "images/val")[:10]
    
    sc, md, fe = load_classification_pipeline()
    
    results = []
    
    for img in val_images:
        res = analyze_blood_smear(root / "images/val" / img, sc, md, fe)
        c = res["counts"]
        
        # Load GT
        lbl_path = root / "labels/val" / img.replace(".png", ".txt").replace(".jpg", ".txt")
        gt_total = 0
        if lbl_path.exists():
            with open(lbl_path) as f:
                gt_total = len([l for l in f if l.strip()])
                
        results.append({
            "image": img,
            "pred_total": c["Total"],
            "pred_RBC": c["RBC"],
            "pred_WBC": c["WBC"],
            "pred_Platelets": c["Platelets"],
            "gt_total": gt_total,
            "MAE": abs(c["Total"] - gt_total)
        })
        
    df = pd.DataFrame(results)
    print(df)
    
    out_dir = Path("results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "segmentation_regression.csv", index=False)
    print("Saved segmentation_regression.csv")

if __name__ == "__main__":
    validate_pipeline()
