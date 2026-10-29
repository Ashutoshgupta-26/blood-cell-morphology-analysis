import os
from pathlib import Path
from src.counting import load_classification_pipeline, analyze_blood_smear
from config import locate_dataset_root

def test_regression():
    root = locate_dataset_root()
    imgs = os.listdir(root / "images/test")[:5]
    scaler, model, features = load_classification_pipeline()
    
    print("Regression Test Results:")
    for img in imgs:
        img_path = root / "images/test" / img
        res = analyze_blood_smear(img_path, scaler, model, features)
        c = res["counts"]
        print(f"Image: {img} -> Instances: {c['Total']} (RBC:{c['RBC']}, WBC:{c['WBC']}, PLT:{c['Platelets']})")

if __name__ == "__main__":
    test_regression()
