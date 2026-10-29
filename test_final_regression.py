"""Final 10-image regression across test set, including the problem image."""
import cv2, numpy as np, os, pandas as pd
from src.preprocessing import load_image
from src.counting import load_classification_pipeline, analyze_blood_smear
from config import locate_dataset_root

root = locate_dataset_root()
test_images = sorted(os.listdir(root / "images/test"))[:10]

# Make sure problem image is included
problem = "7f96e833a77559fb8fe76152b2b76660.png"
if problem not in test_images:
    test_images[-1] = problem

s, m, f = load_classification_pipeline()

rows = []
for img_name in test_images:
    img_path = root / "images/test" / img_name
    lbl_path = root / "labels/test" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
    if not lbl_path.exists(): continue
    
    img = load_image(img_path)
    h_img, w_img = img.shape[:2]
    
    # GT
    gt_boxes = []
    with open(lbl_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                gt_boxes.append(int(parts[0]))
    
    gt_rbc = sum(1 for c in gt_boxes if c == 1)
    gt_wbc = sum(1 for c in gt_boxes if c == 2)
    gt_plt = sum(1 for c in gt_boxes if c == 0)
    gt_total = len(gt_boxes)
    
    # Predict
    r = analyze_blood_smear(str(img_path), s, m, f)
    counts = r['counts']
    
    rows.append({
        "Image": img_name[:16] + "...",
        "Pred_RBC": counts.get('RBC', 0),
        "GT_RBC": gt_rbc,
        "Pred_WBC": counts.get('WBC', 0),
        "GT_WBC": gt_wbc,
        "Pred_PLT": counts.get('Platelets', 0),
        "GT_PLT": gt_plt,
        "Pred_Total": counts.get('Total', 0),
        "GT_Total": gt_total,
        "MAE": abs(counts.get('Total', 0) - gt_total)
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nMean MAE: {df['MAE'].mean():.1f}")
print(f"Mean Pred Total: {df['Pred_Total'].mean():.1f}")
print(f"Mean GT Total: {df['GT_Total'].mean():.1f}")
df.to_csv("results/debug/rbc_fix/v2/final_regression.csv", index=False)
