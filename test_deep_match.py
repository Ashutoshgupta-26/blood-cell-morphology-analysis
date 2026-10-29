"""
Deep analysis: match each predicted instance to the closest GT box.
Identify false positives, false negatives, and misclassifications.
"""
import cv2, numpy as np
from src.preprocessing import load_image
from src.counting import load_classification_pipeline, analyze_blood_smear
from src.segmentation import segment_cells
from src.watershed import separate_touching_cells
from config import locate_dataset_root

root = locate_dataset_root()
img_path = root / "images/test/7f96e833a77559fb8fe76152b2b76660.png"
lbl_path = root / "labels/test/7f96e833a77559fb8fe76152b2b76660.txt"
img = load_image(img_path)
h_img, w_img = img.shape[:2]

CLASS_MAP = {0: 'WBC', 1: 'RBC', 2: 'Platelets'}

# Load GT boxes
gt_boxes = []
with open(lbl_path) as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if not parts: continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = max(0, int((cx - w/2) * w_img))
        y1 = max(0, int((cy - h/2) * h_img))
        x2 = min(w_img, int((cx + w/2) * w_img))
        y2 = min(h_img, int((cy + h/2) * h_img))
        gt_boxes.append({
            'id': i+1, 'cls': CLASS_MAP[cls_id],
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cx': int(cx * w_img), 'cy': int(cy * h_img)
        })

# Run pipeline
s, m, fc = load_classification_pipeline()
r = analyze_blood_smear(str(img_path), s, m, fc)
df = r['predictions_df']
labels = r['labels']

# For each prediction, compute centroid
pred_instances = []
for _, row in df.iterrows():
    lid = int(row['instance_id'])
    mask = (labels == lid).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        M = cv2.moments(cnts[0])
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0
    else:
        cx, cy = 0, 0
    pred_instances.append({
        'id': lid, 'cls': row['predicted_class'],
        'cx': cx, 'cy': cy,
        'area': row['area'], 'circ': row['circularity'], 'sol': row['solidity']
    })

# Match predictions to GT
matched_gt = set()
matched_pred = set()
matches = []

for pi in pred_instances:
    best_dist = 999999
    best_gt = None
    for gt in gt_boxes:
        if gt['id'] in matched_gt: continue
        dist = np.sqrt((pi['cx'] - gt['cx'])**2 + (pi['cy'] - gt['cy'])**2)
        if dist < best_dist:
            best_dist = dist
            best_gt = gt
    
    # Match if within 50px
    if best_gt and best_dist < 50:
        matches.append((pi, best_gt, best_dist))
        matched_gt.add(best_gt['id'])
        matched_pred.add(pi['id'])

# Report
print("=" * 80)
print("MATCHED PREDICTIONS (True Positives)")
print("=" * 80)
print(f"{'Pred':>5} {'P_Cls':>8} {'GT':>4} {'G_Cls':>8} {'Dist':>6} {'Area':>7} {'Circ':>6} {'Sol':>6} {'Match':>6}")
for pi, gt, dist in sorted(matches, key=lambda x: x[0]['id']):
    ok = "OK" if pi['cls'] == gt['cls'] else "MIS"
    print(f"{pi['id']:5d} {pi['cls']:>8} {gt['id']:4d} {gt['cls']:>8} {dist:6.1f} {pi['area']:7.0f} {pi['circ']:6.2f} {pi['sol']:6.2f} {ok:>6}")

print(f"\n{'=' * 80}")
print("FALSE POSITIVES (predicted but no GT match)")
print("=" * 80)
for pi in pred_instances:
    if pi['id'] not in matched_pred:
        print(f"  Pred #{pi['id']} = {pi['cls']} @ ({pi['cx']},{pi['cy']}) area={pi['area']:.0f} circ={pi['circ']:.2f} sol={pi['sol']:.2f}")

print(f"\n{'=' * 80}")
print("FALSE NEGATIVES (GT box not matched)")
print("=" * 80)
for gt in gt_boxes:
    if gt['id'] not in matched_gt:
        print(f"  GT #{gt['id']} = {gt['cls']} @ ({gt['cx']},{gt['cy']}) box=[{gt['x1']},{gt['y1']},{gt['x2']},{gt['y2']}]")

print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)
correct_cls = sum(1 for pi, gt, _ in matches if pi['cls'] == gt['cls'])
print(f"True Positives:     {len(matches)}")
print(f"  Correct class:    {correct_cls}")
print(f"  Misclassified:    {len(matches) - correct_cls}")
print(f"False Positives:    {len(pred_instances) - len(matched_pred)}")
print(f"False Negatives:    {len(gt_boxes) - len(matched_gt)}")
print(f"GT Total:           {len(gt_boxes)}")
print(f"Predicted Total:    {len(pred_instances)}")
