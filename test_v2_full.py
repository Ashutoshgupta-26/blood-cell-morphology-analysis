"""V2 Full Visualization: Annotated output + per-box GT coverage comparison."""
import cv2
import numpy as np
from src.preprocessing import load_image
from src.counting import load_classification_pipeline, analyze_blood_smear
from src.segmentation import segment_cells
from src.watershed import separate_touching_cells
from config import locate_dataset_root

root = locate_dataset_root()
img_path = root / "images/test/7f96e833a77559fb8fe76152b2b76660.png"
lbl_path = root / "labels/test/7f96e833a77559fb8fe76152b2b76660.txt"
img = load_image(img_path)

# --- Run full pipeline ---
s, m, f = load_classification_pipeline()
results = analyze_blood_smear(str(img_path), s, m, f)
df = results['predictions_df']
labels = results['labels']

# --- Draw annotated image ---
colors = {'RBC': (0, 255, 0), 'WBC': (255, 0, 0), 'Platelets': (0, 255, 255)}
vis = img.copy()
for _, row in df.iterrows():
    lid = int(row['instance_id'])
    cls = row['predicted_class']
    mask = (labels == lid)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    col = colors.get(cls, (128, 128, 128))
    cv2.drawContours(vis, cnts, -1, col, 2)
    if cnts:
        M = cv2.moments(cnts[0])
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(vis, f"{cls[0]}{lid}", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

cv2.imwrite("results/debug/rbc_fix/v2/final_v2_annotated.png", vis)

# --- Segmentation stage debug ---
seg = segment_cells(img, method='triangle', channel='green')
cv2.imwrite("results/debug/rbc_fix/v2/final_v2_raw_mask.png", seg['raw_mask'])
cv2.imwrite("results/debug/rbc_fix/v2/final_v2_cleaned_mask.png", seg['cleaned_mask'])

ws = separate_touching_cells(seg['cleaned_mask'], img)
cv2.imwrite("results/debug/rbc_fix/v2/final_v2_watershed_overlay.png", ws['debug_overlay'])

# --- GT coverage ---
h_img, w_img = img.shape[:2]
boxes = []
with open(lbl_path) as fh:
    for line in fh:
        parts = line.strip().split()
        if not parts: continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = max(0, int((cx - w/2) * w_img))
        y1 = max(0, int((cy - h/2) * h_img))
        x2 = min(w_img, int((cx + w/2) * w_img))
        y2 = min(h_img, int((cy + h/2) * h_img))
        boxes.append((cls_id, x1, y1, x2, y2))

cleaned = seg['cleaned_mask']
print(f"\nGT Coverage (Triangle V2):")
print(f"{'GT#':>4} {'Class':>6} {'Coverage':>8}")
for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
    area = (x2 - x1) * (y2 - y1)
    if area == 0: continue
    cov = np.count_nonzero(cleaned[y1:y2, x1:x2] == 255) / area
    cls_name = ['Platelets', 'RBC', 'WBC'][cls_id] if cls_id < 3 else '?'
    print(f"{i+1:4d} {cls_name:>6} {cov:8.3f}")

covs = [np.count_nonzero(cleaned[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for _, x1, y1, x2, y2 in boxes if (x2-x1)*(y2-y1) > 0]
print(f"\nMean Coverage: {np.mean(covs):.3f}")
print(f"Boxes with Coverage < 0.3: {sum(1 for c in covs if c < 0.3)}")
print(f"Boxes with Coverage > 0.6: {sum(1 for c in covs if c > 0.6)}")

print(f"\nFinal Counts: {results['counts']}")
print(f"GT Boxes: {len(boxes)}")
