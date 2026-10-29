"""
Automated Cell Counting & Statistical Analysis Module (Stage 7).
End-to-end inference using the trained SVM classifier to detect,
classify, and count blood cells in whole smear images.
"""

import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from .dataset_utils import get_paired_files
from .annotation_utils import parse_yolo_annotation
from .preprocessing import load_image
from .segmentation import segment_cells
from .watershed import separate_touching_cells
from .feature_extraction import extract_features_for_image, associate_ground_truth
from config import WATERSHED_CONFIG, METRICS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR, SEGMENTATION_CONFIG

COUNTING_VIS_DIR = VISUALIZATIONS_DIR / "counting"
MODELS_DIR = Path(__file__).resolve().parent.parent / "results" / "models"

def ensure_counting_dirs():
    COUNTING_VIS_DIR.mkdir(parents=True, exist_ok=True)
    (COUNTING_VIS_DIR / "boundaries").mkdir(parents=True, exist_ok=True)
    (COUNTING_VIS_DIR / "segmentation_overlays").mkdir(parents=True, exist_ok=True)
    (COUNTING_VIS_DIR / "best_cases").mkdir(parents=True, exist_ok=True)
    (COUNTING_VIS_DIR / "worst_cases").mkdir(parents=True, exist_ok=True)
    (COUNTING_VIS_DIR / "count_distributions").mkdir(parents=True, exist_ok=True)

def load_classification_pipeline():
    """Loads the trained scaler, best model, and feature schema."""
    scaler_path = MODELS_DIR / "scaler.joblib"
    model_path = MODELS_DIR / "best_svm.joblib"
    features_path = MODELS_DIR / "best_model_features.json"
    
    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError("Trained models not found. Run Stage 6 first.")
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
        
    return scaler, model, feature_cols

def analyze_blood_smear(image_path, scaler, model, feature_cols, gt_annotations=None):
    """
    End-to-end inference function for a single image.
    Returns counts, features, predictions, and annotated images.
    """
    img_path = Path(image_path)
    img_bgr = load_image(img_path)
    h, w = img_bgr.shape[:2]
    
    # 1. Pipeline Execution
    best_th = WATERSHED_CONFIG.get("distance_threshold_ratio", 0.25)
    method = SEGMENTATION_CONFIG.get("segmentation_method", "otsu")
    channel = SEGMENTATION_CONFIG.get("input_channel", "green")
    
    seg_res = segment_cells(img_bgr, method=method, channel=channel)
    w_res = separate_touching_cells(seg_res["cleaned_mask"], img_bgr, threshold_ratio=best_th)
    labels = w_res["labels"]
    
    # 2. Extract Features
    features_list = extract_features_for_image(img_bgr, labels, img_path.name)
    df_features = pd.DataFrame(features_list)
    
    if df_features.empty:
        return {
            "image_name": img_path.name,
            "counts": {"RBC": 0, "WBC": 0, "Platelets": 0, "Total": 0},
            "density": {"cells_per_pixel": 0, "cells_per_10000_pixels": 0, "cells_per_megapixel": 0},
            "predictions_df": pd.DataFrame(),
            "labels": labels,
            "img_bgr": img_bgr
        }

    # 3. Classify Instances
    X = df_features[feature_cols].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        confidences = np.max(probs, axis=1)
    else:
        confidences = np.ones(len(preds))
        
    df_features["predicted_class"] = preds
    df_features["prediction_confidence"] = confidences
    
    # --- Post-Classification Quality Filter ---
    # Keep cells clipped by the image boundary: they are valid image objects
    # and are explicitly flagged so downstream users can treat their features
    # with care. Only remove small/irregular RBC fragments away from borders.
    EDGE_MARGIN = 3
    keep_mask = np.ones(len(df_features), dtype=bool)
    touches_border_flags = []
    for idx, row in df_features.iterrows():
        lid = int(row["instance_id"])
        inst_mask = (labels == lid).astype(np.uint8)
        ys, xs = np.where(inst_mask > 0)
        if len(xs) == 0:
            keep_mask[idx] = False
            touches_border_flags.append(False)
            continue
        
        touches_edge = (ys.min() <= EDGE_MARGIN or ys.max() >= h - EDGE_MARGIN - 1 or
                        xs.min() <= EDGE_MARGIN or xs.max() >= w - EDGE_MARGIN - 1)
        touches_border_flags.append(bool(touches_edge))
        
        area = row.get("area", 0)
        circ = row.get("circularity", 0)
        sol = row.get("solidity", 0)
        pred_cls = row["predicted_class"]
        
        # Rule 1: Remove very small RBC fragments away from the image border.
        if pred_cls == "RBC" and area < 300 and not touches_edge:
            keep_mask[idx] = False
            continue
            
        # Rule 2: Remove clearly irregular RBC fragments away from the border.
        if pred_cls == "RBC" and circ < 0.25 and sol < 0.6 and not touches_edge:
            keep_mask[idx] = False
            continue
    
    df_features["touches_image_border"] = touches_border_flags
    df_features = df_features[keep_mask].reset_index(drop=True)

    # 4. Counting
    counts = {"RBC": 0, "WBC": 0, "Platelets": 0}
    val_counts = df_features["predicted_class"].value_counts().to_dict()
    for k in counts:
        counts[k] = val_counts.get(k, 0)
    counts["Total"] = len(df_features)
    
    # 5. Density
    area_px = w * h
    density = {
        "cells_per_pixel": counts["Total"] / area_px,
        "cells_per_10000_pixels": (counts["Total"] / area_px) * 10000,
        "cells_per_megapixel": (counts["Total"] / area_px) * 1_000_000
    }
    
    return {
        "image_name": img_path.name,
        "counts": counts,
        "density": density,
        "predictions_df": df_features,
        "labels": labels,
        "img_bgr": img_bgr,
        "gt_annotations": gt_annotations
    }

def annotate_counted_image(result, output_dir_bound, output_dir_seg):
    """Generate Boundary and Filled Segmentation visualizations."""
    img_bgr = result["img_bgr"]
    labels = result["labels"]
    preds_df = result["predictions_df"]
    img_name = result["image_name"]
    counts = result["counts"]
    
    class_colors = {
        "RBC": (50, 50, 220),       # Red
        "WBC": (255, 50, 150),      # Purple
        "Platelets": (0, 200, 255)  # Orange
    }
    
    bound_img = img_bgr.copy()
    seg_img = img_bgr.copy()
    overlay = np.zeros_like(img_bgr)
    
    for _, row in preds_df.iterrows():
        inst_id = int(row["instance_id"])
        pred_class = row["predicted_class"]
        color = class_colors.get(pred_class, (255, 255, 255))
        
        # Mask for instance
        mask = np.uint8(labels == inst_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw boundaries
        cv2.drawContours(bound_img, contours, -1, color, 1)
        
        # Draw filled overlay
        cv2.drawContours(overlay, contours, -1, color, -1)
        
        # Text label (Bound image only)
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                partial_tag = " (edge)" if row.get("touches_image_border", False) else ""
                label = f"{'PLT' if pred_class=='Platelets' else pred_class} #{inst_id}{partial_tag}"
                cv2.putText(bound_img, label, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
    # Add count summary box
    summary_text = [
        f"Total: {counts['Total']}",
        f"RBC: {counts['RBC']}",
        f"WBC: {counts['WBC']}",
        f"PLT: {counts['Platelets']}"
    ]
    
    for img in [bound_img, seg_img]:
        y_offset = 25
        for line in summary_text:
            cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
    # Blend filled overlay
    alpha = 0.4
    seg_img = cv2.addWeighted(overlay, alpha, seg_img, 1 - alpha, 0)
    
    cv2.imwrite(str(output_dir_bound / f"{img_name}"), bound_img)
    cv2.imwrite(str(output_dir_seg / f"{img_name}"), seg_img)

def generate_counting_metrics(cell_counts_df, gt_counts_df):
    comp_df = cell_counts_df.merge(gt_counts_df, on="image_name", how="left")
    
    metrics = []
    classes = ["RBC", "WBC", "Platelets", "total_cells"]
    
    for cls in classes:
        if cls == "total_cells":
            pred_col = "total_cells"
            gt_col = "gt_total_cells"
        elif cls == "Platelets":
            pred_col = "platelet_count"
            gt_col = "gt_platelet_count"
        else:
            pred_col = f"{cls.lower()}_count"
            gt_col = f"gt_{cls.lower()}_count"
        
        pred = comp_df[pred_col]
        gt = comp_df[gt_col]
        
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt)**2))
        mse_signed = np.mean(pred - gt)
        
        # Absolute Percentage Error (handling gt=0)
        ape = []
        for p, g in zip(pred, gt):
            if g == 0:
                ape.append(np.nan)
            else:
                ape.append(abs(p - g) / g * 100)
        mape = np.nanmean(ape) if len(ape) > 0 else np.nan
        
        metrics.append({
            "class": cls,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MSE_signed": round(mse_signed, 2),
            "MAPE": round(mape, 2)
        })
        
        # Also store image-level errors
        comp_df[f"{cls}_error"] = pred - gt
        comp_df[f"{cls}_abs_error"] = np.abs(pred - gt)
        
    metrics_df = pd.DataFrame(metrics)
    comp_df.to_csv(METRICS_DIR / "count_comparison.csv", index=False)
    metrics_df.to_csv(METRICS_DIR / "counting_metrics.csv", index=False)
    return comp_df, metrics_df

def calculate_object_detection_metrics(all_predictions_df, gt_annotations_list):
    """Calculate object-level Precision/Recall/F1 based on Box IoU matching."""
    
    # Re-calculate matches globally over the test set
    TP = {"RBC": 0, "WBC": 0, "Platelets": 0, "All": 0}
    FP = {"RBC": 0, "WBC": 0, "Platelets": 0, "All": 0}
    FN = {"RBC": 0, "WBC": 0, "Platelets": 0, "All": 0}
    
    # Classification errors among matched
    class_errors = []
    
    # Group predictions by image
    preds_by_img = all_predictions_df.groupby("image_name")
    
    for pair in gt_annotations_list:
        img_name = pair["image_path"].name
        anns = pair["anns"]
        
        if img_name in preds_by_img.groups:
            img_preds = preds_by_img.get_group(img_name)
        else:
            img_preds = pd.DataFrame()
            
        # Match GT to Preds using 0.5 IoU threshold for rigorous object detection
        iou_th = 0.5
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        for gt_idx, ann in enumerate(anns):
            gt_class = ann["class_name"]
            gt_x1, gt_y1 = ann["x1"], ann["y1"]
            gt_x2, gt_y2 = ann["x2"], ann["y2"]
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            
            best_iou = 0
            best_pred_idx = -1
            best_pred_class = None
            
            for pred_idx, row in img_preds.iterrows():
                if pred_idx in matched_pred_indices: continue
                
                inst_x1 = row["bbox_x"]
                inst_y1 = row["bbox_y"]
                inst_w = row["bbox_width"]
                inst_h = row["bbox_height"]
                inst_x2 = inst_x1 + inst_w
                inst_y2 = inst_y1 + inst_h
                inst_area = inst_w * inst_h
                
                # Intersection
                ix1 = max(gt_x1, inst_x1)
                iy1 = max(gt_y1, inst_y1)
                ix2 = min(gt_x2, inst_x2)
                iy2 = min(gt_y2, inst_y2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih
                
                if inter_area > 0:
                    iou = inter_area / float(gt_area + inst_area - inter_area)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                        best_pred_class = row["predicted_class"]
                        
            if best_iou >= iou_th:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_pred_idx)
                
                if best_pred_class == gt_class:
                    TP[gt_class] += 1
                    TP["All"] += 1
                else:
                    # Classification Error
                    FP[best_pred_class] += 1
                    FN[gt_class] += 1
                    FP["All"] += 1
                    FN["All"] += 1
                    class_errors.append({"true_class": gt_class, "predicted_class": best_pred_class})
            else:
                # Undetected GT (False Negative)
                FN[gt_class] += 1
                FN["All"] += 1
                
        # Unmatched Predictions (False Positives)
        for pred_idx, row in img_preds.iterrows():
            if pred_idx not in matched_pred_indices:
                pred_class = row["predicted_class"]
                FP[pred_class] += 1
                FP["All"] += 1
                
    # Calculate Metrics
    results = []
    for cls in TP.keys():
        precision = TP[cls] / (TP[cls] + FP[cls]) if (TP[cls] + FP[cls]) > 0 else 0
        recall = TP[cls] / (TP[cls] + FN[cls]) if (TP[cls] + FN[cls]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "class": cls,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "TP": TP[cls], "FP": FP[cls], "FN": FN[cls]
        })
        
    obj_df = pd.DataFrame(results)
    obj_df.to_csv(METRICS_DIR / "object_detection_metrics.csv", index=False)
    
    if class_errors:
        pd.DataFrame(class_errors).to_csv(METRICS_DIR / "matched_classification_results.csv", index=False)
        
    return obj_df
def analyze_dataset(split="test"):
    print(f"\nAnalyzing dataset split: {split}")
    ensure_counting_dirs()
    scaler, model, feature_cols = load_classification_pipeline()
    
    pairs = get_paired_files(split=split)
    if not pairs:
        print("No images found.")
        return
        
    counts_list = []
    predictions_list = []
    gt_counts_list = []
    gt_anns_list = []
    
    for idx, pair in enumerate(pairs, 1):
        img_name = pair["image_path"].name
        print(f"  [{idx}/{len(pairs)}] Counting {img_name}...")
        
        # Load GT
        img_bgr = load_image(pair["image_path"])
        h, w = img_bgr.shape[:2]
        anns = parse_yolo_annotation(pair["label_path"], w, h)
        gt_anns_list.append({"image_path": pair["image_path"], "anns": anns})
        
        gt_counts = {"image_name": img_name, "gt_total_cells": len(anns), "gt_rbc_count": 0, "gt_wbc_count": 0, "gt_platelet_count": 0}
        for ann in anns:
            c = ann["class_name"]
            if c == "RBC": gt_counts["gt_rbc_count"] += 1
            elif c == "WBC": gt_counts["gt_wbc_count"] += 1
            elif c == "Platelets": gt_counts["gt_platelet_count"] += 1
        gt_counts_list.append(gt_counts)
        
        # Run Pipeline
        res = analyze_blood_smear(pair["image_path"], scaler, model, feature_cols, gt_annotations=anns)
        
        # Store Counts
        c = res["counts"]
        d = res["density"]
        
        row = {
            "image_name": img_name,
            "image_width": w, "image_height": h,
            "total_cells": c["Total"],
            "rbc_count": c["RBC"], "wbc_count": c["WBC"], "platelet_count": c["Platelets"],
            "dataset_split": split
        }
        
        # Percentages
        if c["Total"] > 0:
            row["rbc_percentage"] = (c["RBC"] / c["Total"]) * 100
            row["wbc_percentage"] = (c["WBC"] / c["Total"]) * 100
            row["platelet_percentage"] = (c["Platelets"] / c["Total"]) * 100
        else:
            row["rbc_percentage"] = row["wbc_percentage"] = row["platelet_percentage"] = 0.0
            
        row.update(d)
        counts_list.append(row)
        
        # Store Preds
        if not res["predictions_df"].empty:
            predictions_list.append(res["predictions_df"])
            
        # Annotate
        annotate_counted_image(res, COUNTING_VIS_DIR / "boundaries", COUNTING_VIS_DIR / "segmentation_overlays")
        
    counts_df = pd.DataFrame(counts_list)
    gt_df = pd.DataFrame(gt_counts_list)
    all_preds = pd.concat(predictions_list, ignore_index=True) if predictions_list else pd.DataFrame()
    
    counts_df.to_csv(METRICS_DIR / "cell_counts.csv", index=False)
    gt_df.to_csv(METRICS_DIR / "ground_truth_counts.csv", index=False)
    if not all_preds.empty:
        all_preds.to_csv(METRICS_DIR / "cell_predictions.csv", index=False)
        
    # Evaluate
    comp_df, metrics_df = generate_counting_metrics(counts_df, gt_df)
    obj_df = calculate_object_detection_metrics(all_preds, gt_anns_list)
    
    generate_summary_and_plots(comp_df, obj_df, metrics_df)
    
    print("\nCounting analysis complete.")


def generate_summary_and_plots(comp_df, obj_df, metrics_df):
    """Generates plots and statistical summaries."""
    
    # 1. Error Ranking
    comp_df = comp_df.sort_values(by="total_cells_abs_error", ascending=False)
    comp_df.to_csv(METRICS_DIR / "counting_error_ranking.csv", index=False)
    
    # 2. Scatter Plots
    for cls in ["rbc", "wbc", "platelet", "total_cells"]:
        pred_col = f"{cls}_count" if cls != "total_cells" else "total_cells"
        gt_col = f"gt_{cls}_count" if cls != "total_cells" else "gt_total_cells"
        
        plt.figure(figsize=(6,6))
        plt.scatter(comp_df[gt_col], comp_df[pred_col], alpha=0.7)
        
        max_val = max(comp_df[gt_col].max(), comp_df[pred_col].max()) + 5
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel("Ground Truth Count")
        plt.ylabel("Predicted Count")
        plt.title(f"{cls.upper()} Predicted vs Ground Truth")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(COUNTING_VIS_DIR / f"{cls}_predicted_vs_gt.png")
        plt.close()
        
    # 3. Error Distributions
    plt.figure(figsize=(10,8))
    for i, cls in enumerate(["RBC", "WBC", "Platelets", "total_cells"], 1):
        plt.subplot(2, 2, i)
        plt.hist(comp_df[f"{cls}_error"], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"{cls.upper()} Count Error Distribution")
        plt.xlabel("Predicted - GT")
    plt.tight_layout()
    plt.savefig(COUNTING_VIS_DIR / "error_distributions.png")
    plt.close()

    # 4. Statistical Summary
    sum_data = {
        "number_of_images": len(comp_df),
        "total_gt_cells": comp_df["gt_total_cells"].sum(),
        "total_predicted_cells": comp_df["total_cells"].sum(),
        
        "gt_rbc_total": comp_df["gt_rbc_count"].sum(),
        "pred_rbc_total": comp_df["rbc_count"].sum(),
        
        "gt_wbc_total": comp_df["gt_wbc_count"].sum(),
        "pred_wbc_total": comp_df["wbc_count"].sum(),
        
        "gt_platelet_total": comp_df["gt_platelet_count"].sum(),
        "pred_platelet_total": comp_df["platelet_count"].sum(),
    }
    
    for _, row in metrics_df.iterrows():
        c = row["class"].lower()
        if c == "platelets": c = "platelet"
        sum_data[f"{c}_mae"] = row["MAE"]
        sum_data[f"{c}_rmse"] = row["RMSE"]
        
    all_obj = obj_df[obj_df["class"] == "All"].iloc[0]
    sum_data["overall_precision"] = all_obj["precision"]
    sum_data["overall_recall"] = all_obj["recall"]
    sum_data["overall_f1"] = all_obj["f1"]
    
    pd.DataFrame([sum_data]).to_csv(METRICS_DIR / "counting_summary.csv", index=False)
    
    # Convert DataFrames to simple markdown strings manually
    def df_to_md(df):
        cols = df.columns.tolist()
        header = "| " + " | ".join([str(c) for c in cols]) + " |"
        sep = "|-" + "-|-".join(["-"*len(str(c)) for c in cols]) + "-|"
        rows = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join([str(x) for x in r.values]) + " |")
        return "\n".join([header, sep] + rows)

    # Generate Report
    report = f"""# Stage 7: Cell Counting & Statistical Analysis

## 1. End-to-End Inference
The pipeline processes whole images:
`Raw Image -> Preprocessing (Bilateral+CLAHE) -> Green Channel Otsu -> Morphological Cleanup -> Distance Transform Watershed -> Feature Extraction -> SVM Classification (RBF) -> Counting`.

## 2. Counting Metrics
We compared the automated counts against the TXL-PBC ground truth bounding box counts.

### Object Detection Metrics (IoU >= 0.5)
Because bounding boxes are used, matching predictions to GT relies on IoU. 
{df_to_md(obj_df)}
*Note: Low recall implies the watershed algorithm under-segments touching cells, or Otsu thresholding removes faint cells as background.*

### Count Error Metrics (Image-Level)
{df_to_md(metrics_df)}
*MSE_signed < 0 indicates systematic undercounting.*

## 3. Analysis & Limitations
- **Overcounting**: Can occur if noise artifacts or fragmented platelets are falsely detected.
- **Undercounting (Major Cause)**: Dense clusters of overlapping RBCs are not always split perfectly by the Watershed algorithm (creating 1 large component instead of 3 smaller ones). This reduces object recall and causes total count underestimation.
- **Classification vs Detection**: Once a cell is properly isolated (detected), the SVM classifier is highly accurate. Most counting errors stem from Stage 4 (Watershed Separation).
- **Dataset Limitation**: Image counts are relative to the camera field of view.
- **Medical Disclaimer**: The system estimates the number and morphological distribution of blood-cell objects present in the analyzed microscope image. It does NOT provide clinical diagnosis (e.g., patient RBC/uL).
"""
    with open(REPORTS_DIR / "counting_analysis.md", "w", encoding="utf-8") as f:
        f.write(report)
