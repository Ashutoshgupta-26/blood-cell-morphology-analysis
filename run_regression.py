import pandas as pd
from pathlib import Path
from src.counting import load_classification_pipeline, analyze_blood_smear
from config import locate_dataset_root
import os

def run_regression_test():
    root = locate_dataset_root()
    
    # 6 images to test
    imgs = [
        "00552f60c43350a0bf516cfbc13db058.png",
        "00bb17f0c091599da3f97aeeef49bec6.png",
        "02b9abe156065d8097edfd6afbebcbb3.png",
        "03967e3a34d05f4583e6d361f2c333a0.png",
        "03e9e4f8980a5fe78e1069167ff9f3d9.png",
        "7f96e833a77559fb8fe76152b2b76660.png"
    ]
    
    scaler, model, features = load_classification_pipeline()
    results = []
    
    for img in imgs:
        img_path = root / "images/test" / img
        if not img_path.exists():
            continue
            
        res = analyze_blood_smear(img_path, scaler, model, features)
        df = res["predictions_df"]
        
        # Calculate large/small fragment count based on the new logic
        large_fragments = len(df[(df["predicted_class"] == "RBC") & (df["area"] < 1000)])
        small_objects = len(df[df["area"] < 300])
        
        c = res["counts"]
        results.append({
            "image_name": img,
            "instances": c["Total"],
            "RBC": c["RBC"],
            "WBC": c["WBC"],
            "platelets": c["Platelets"],
            "large_fragment_count": large_fragments,
            "small_object_count": small_objects
        })
        
    df_res = pd.DataFrame(results)
    out_dir = Path("results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "regression_test.csv", index=False)
    print("Saved regression_test.csv")

if __name__ == "__main__":
    run_regression_test()
