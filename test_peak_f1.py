"""Evaluate different min_peak_distance values with full GT matching."""
import cv2, numpy as np, os
from src.preprocessing import load_image
from src.segmentation import segment_cells
from src.watershed import compute_distance_transform, generate_foreground_markers, generate_background_markers
from src.counting import load_classification_pipeline
from config import locate_dataset_root, WATERSHED_CONFIG

CLASS_MAP = {0: 'WBC', 1: 'RBC', 2: 'Platelets'}
root = locate_dataset_root()

# Load classifier
scaler, model, feature_cols = load_classification_pipeline()

# Gather 10 val images
val_images = sorted(os.listdir(root / "images/val"))[:10]

for min_d in [10, 12, 15]:
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for img_name in val_images:
        img_path = root / "images/val" / img_name
        lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        if not lbl_path.exists(): continue
        
        img = load_image(img_path)
        h_img, w_img = img.shape[:2]
        
        # GT
        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cx, cy, w, h = map(float, parts[1:5])
                gt_boxes.append((int(cx * w_img), int(cy * h_img)))
        
        # Segmentation + watershed with custom min_d
        seg = segment_cells(img)
        dt = compute_distance_transform(seg['cleaned_mask'])
        sf, markers, nl = generate_foreground_markers(dt, seg['cleaned_mask'], min_distance=min_d)
        sb = generate_background_markers(seg['cleaned_mask'])
        unknown = cv2.subtract(sb, sf)
        mc = markers.copy() + 1
        mc[unknown == 255] = 0
        mw = cv2.watershed(img.copy(), mc)
        
        # Count valid instances
        min_area = WATERSHED_CONFIG.get("min_cell_area", 100)
        pred_centroids = []
        for lid in range(2, nl + 1):
            mask = (mw == lid)
            if np.count_nonzero(mask) < min_area: continue
            ys, xs = np.where(mask)
            pred_centroids.append((int(np.mean(xs)), int(np.mean(ys))))
        
        # Match
        matched_gt = set()
        for pcx, pcy in pred_centroids:
            best_dist = 999999
            best_idx = -1
            for i, (gcx, gcy) in enumerate(gt_boxes):
                if i in matched_gt: continue
                d = np.sqrt((pcx - gcx)**2 + (pcy - gcy)**2)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0 and best_dist < 50:
                matched_gt.add(best_idx)
                tp_total += 1
            else:
                fp_total += 1
        
        fn_total += len(gt_boxes) - len(matched_gt)
    
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"min_d={min_d:2d}: TP={tp_total:3d} FP={fp_total:3d} FN={fn_total:3d} Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f}")
