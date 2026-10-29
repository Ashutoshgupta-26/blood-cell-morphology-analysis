import cv2
from src.preprocessing import load_image
from src.counting import analyze_blood_smear
from config import locate_dataset_root

def test_final_v2():
    root = locate_dataset_root()
    img_path = root / "images/test/7f96e833a77559fb8fe76152b2b76660.png"
    img = load_image(img_path)
    
    from src.counting import load_classification_pipeline
    scaler, model, feature_cols = load_classification_pipeline()
    
    results = analyze_blood_smear(
        str(img_path),
        scaler=scaler,
        model=model,
        feature_cols=feature_cols
    )
    print("V2 Metrics:")
    print("Total Cells:", results["metrics"]["total_cells"])
    print("RBC:", results["metrics"]["rbc_count"])
    print("WBC:", results["metrics"]["wbc_count"])
    print("Platelets:", results["metrics"]["platelet_count"])
    
    cv2.imwrite("results/debug/rbc_fix/v2/final_v2_output.png", results["annotated_img"])
    cv2.imwrite("results/debug/rbc_fix/v2/final_v2_watershed.png", results["watershed"]["debug_overlay"])

if __name__ == "__main__":
    test_final_v2()
