import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from collections import Counter

from src.preprocessing import load_image, grayscale_conversion, bilateral_filter, clahe_enhancement
from src.segmentation import otsu_segmentation, clean_segmentation_mask
from src.watershed import separate_touching_cells, compute_instance_statistics
from src.feature_extraction import extract_features_for_image
from src.counting import load_classification_pipeline
from config import locate_dataset_root

def diagnose(img_name):
    root = locate_dataset_root()
    img_path = root / "images/test" / img_name
    lbl_path = root / "labels/test" / img_name.replace('.png', '.txt').replace('.jpg', '.txt')
    
    out_dir = Path("results/debug/problem_image")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("DIAGNOSIS FOR:", img_name)
    print("="*50)
    
    # GT processing
    gt_counts = Counter()
    with open(root / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]
        
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_name = class_names[int(parts[0])]
                    gt_counts[cls_name] += 1
    
    print("GROUND TRUTH:")
    for k, v in gt_counts.items(): print(f"  {k}: {v}")
    
    # 1. Load Image
    img_bgr = load_image(img_path)
    print(f"\nImage shape: {img_bgr.shape}")
    cv2.imwrite(str(out_dir / "01_original.png"), img_bgr)
    
    # 2. Preprocessing
    green = grayscale_conversion(img_bgr, "green")
    filtered = bilateral_filter(green, d=9, sigma_color=75.0, sigma_space=75.0)
    preprocessed = clahe_enhancement(filtered, clip_limit=2.0, tile_grid_size=(8,8))
    cv2.imwrite(str(out_dir / "02_preprocessed.png"), preprocessed)
    
    # 3. Segmentation
    _, raw_mask = otsu_segmentation(preprocessed)
    cv2.imwrite(str(out_dir / "03_raw_segmentation.png"), raw_mask)
    
    # 4. Cleaned Mask
    cleaned_mask = clean_segmentation_mask(raw_mask)
    cv2.imwrite(str(out_dir / "04_cleaned_segmentation.png"), cleaned_mask)
    
    fg_pixels = np.count_nonzero(cleaned_mask)
    total_pixels = cleaned_mask.size
    print(f"Foreground pixels: {fg_pixels} ({fg_pixels/total_pixels*100:.2f}%)")
    
    # 5. Connected Components Before Watershed
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    # Background is label 0
    areas = sorted([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)], reverse=True)
    print(f"\nComponents before watershed: {num_labels - 1}")
    print(f"Top 5 component areas: {areas[:5]}")
    
    vis_cc = np.zeros_like(img_bgr)
    for i in range(1, num_labels):
        vis_cc[labels_cc == i] = np.random.randint(0, 255, 3)
    cv2.imwrite(str(out_dir / "05_connected_components.png"), vis_cc)
    
    # 6. Watershed
    ws_res = separate_touching_cells(cleaned_mask, img_bgr)
    labels_ws = ws_res["labels"]
    
    vis_dist = cv2.normalize(ws_res["dist_transform"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out_dir / "06_distance_transform.png"), vis_dist)
    
    cv2.imwrite(str(out_dir / "07_markers.png"), ws_res["sure_fg"])
    cv2.imwrite(str(out_dir / "08_watershed_raw.png"), np.uint8((labels_ws > 0) * 255))
    
    print(f"Watershed instances: {labels_ws.max()}")
    instances = np.unique(labels_ws[labels_ws > 0])
    
    # 7. Features
    scaler, model, expected_features = load_classification_pipeline()
    print(f"\nClassifier expected features: {len(expected_features)}")
    
    df_features_list = extract_features_for_image(img_bgr, labels_ws, img_name)
    df_features = pd.DataFrame(df_features_list)
    print(f"Extracted feature rows: {len(df_features)}")
    
    if len(df_features) > 0:
        missing = set(expected_features) - set(df_features.columns)
        print(f"Missing features in extraction: {missing}")
        
        # Predict
        X = df_features[expected_features].values
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        df_features["predicted_class"] = preds
        
        print("\nPredictions:")
        pred_counts = df_features["predicted_class"].value_counts()
        for k, v in pred_counts.items(): print(f"  {k}: {v}")
    
if __name__ == "__main__":
    diagnose("00bb17f0c091599da3f97aeeef49bec6.png")
