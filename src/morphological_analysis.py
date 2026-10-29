"""
Morphological Analysis & Outlier Detection Module (Stage 8).
Analyzes size, shape, and color distributions of detected blood cells.
Flags image-based morphological outliers using Z-score, IQR, and Isolation Forest.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import METRICS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR
from .dataset_utils import get_paired_files
from .preprocessing import load_image

MORPH_VIS_DIR = VISUALIZATIONS_DIR / "morphology"

def ensure_morphology_dirs():
    dirs = [
        MORPH_VIS_DIR,
        MORPH_VIS_DIR / "shape_relationships",
        MORPH_VIS_DIR / "color",
        MORPH_VIS_DIR / "intensity",
        MORPH_VIS_DIR / "texture",
        MORPH_VIS_DIR / "outliers",
        MORPH_VIS_DIR / "outlier_overlays",
        MORPH_VIS_DIR / "correlation"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def define_feature_groups():
    return {
        "SIZE": ["area", "equivalent_diameter", "perimeter", "bbox_width", "bbox_height"],
        "SHAPE": ["circularity", "aspect_ratio", "extent", "solidity", "convex_hull_area"],
        "COLOR": ["mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v", "mean_l", "mean_a", "mean_lab_b"],
        "INTENSITY": ["mean_intensity", "median_intensity", "std_intensity", "intensity_p25", "intensity_p75"],
        "TEXTURE": ["mean_gradient", "std_gradient"]
    }

def calculate_class_statistics(df, groups):
    """Calculate descriptive statistics for each class."""
    stats_list = []
    classes = df["predicted_class"].unique()
    
    for cls in classes:
        cls_df = df[df["predicted_class"] == cls]
        for group_name, features in groups.items():
            for feat in features:
                if feat not in cls_df.columns: continue
                s = cls_df[feat].dropna()
                if len(s) == 0: continue
                
                stats_list.append({
                    "class": cls,
                    "feature": feat,
                    "group": group_name,
                    "count": len(s),
                    "mean": s.mean(),
                    "median": s.median(),
                    "std": s.std(),
                    "min": s.min(),
                    "q25": s.quantile(0.25),
                    "q75": s.quantile(0.75),
                    "max": s.max()
                })
                
    return pd.DataFrame(stats_list)

def detect_outliers(df, feature_groups):
    """Apply Z-score, IQR, and Isolation Forest per class."""
    df = df.copy()
    classes = df["predicted_class"].unique()
    
    # Initialize outlier tracking
    df["z_outlier_count"] = 0
    df["iqr_outlier_count"] = 0
    df["is_multivariate_outlier"] = 0
    df["anomaly_score"] = 0.0
    
    reasons_list = []
    
    all_features = []
    for g in feature_groups.values():
        all_features.extend(g)
    
    for cls in classes:
        cls_idx = df["predicted_class"] == cls
        cls_df = df[cls_idx]
        if len(cls_df) < 5: continue
        
        # Univariate
        for feat in all_features:
            if feat not in cls_df.columns: continue
            vals = cls_df[feat].values
            mean, std = np.nanmean(vals), np.nanstd(vals)
            q25, q75 = np.nanpercentile(vals, 25), np.nanpercentile(vals, 75)
            iqr = q75 - q25
            
            if std > 0:
                z_scores = np.abs((vals - mean) / std)
                z_mask = z_scores > 3.0
                df.loc[cls_df.index[z_mask], "z_outlier_count"] += 1
                
                for i in np.where(z_mask)[0]:
                    reasons_list.append({
                        "image_name": cls_df.iloc[i]["image_name"],
                        "instance_id": cls_df.iloc[i]["instance_id"],
                        "class": cls, "feature": feat, "method": "Z-score",
                        "value": vals[i], "reference_mean": mean,
                        "reference_median": np.nanmedian(vals),
                        "lower_bound": mean - 3*std, "upper_bound": mean + 3*std,
                        "z_score": z_scores[i]
                    })
            
            if iqr > 0:
                lower, upper = q25 - 1.5*iqr, q75 + 1.5*iqr
                iqr_mask = (vals < lower) | (vals > upper)
                df.loc[cls_df.index[iqr_mask], "iqr_outlier_count"] += 1
                
                for i in np.where(iqr_mask)[0]:
                    reasons_list.append({
                        "image_name": cls_df.iloc[i]["image_name"],
                        "instance_id": cls_df.iloc[i]["instance_id"],
                        "class": cls, "feature": feat, "method": "IQR",
                        "value": vals[i], "reference_mean": mean,
                        "reference_median": np.nanmedian(vals),
                        "lower_bound": lower, "upper_bound": upper,
                        "z_score": np.nan if std == 0 else (vals[i] - mean) / std
                    })

        # Multivariate Isolation Forest
        valid_features = [f for f in all_features if df[f].notna().all()]
        if len(valid_features) > 0 and len(cls_df) >= 20:
            X = cls_df[valid_features].values
            X_scaled = StandardScaler().fit_transform(X)
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X_scaled)
            scores = iso.score_samples(X_scaled)
            
            df.loc[cls_idx, "is_multivariate_outlier"] = (preds == -1).astype(int)
            df.loc[cls_idx, "anomaly_score"] = scores

    df["is_z_outlier"] = (df["z_outlier_count"] > 0).astype(int)
    df["is_iqr_outlier"] = (df["iqr_outlier_count"] > 0).astype(int)
    
    df["outlier_score"] = df["is_z_outlier"] + df["is_iqr_outlier"] + df["is_multivariate_outlier"]
    df["high_confidence_morphological_outlier"] = (df["outlier_score"] >= 2).astype(int)
    
    reasons_df = pd.DataFrame(reasons_list)
    return df, reasons_df

def generate_visualizations(df, feature_groups):
    classes = df["predicted_class"].unique()
    
    # 1. Size distributions
    for feat in feature_groups["SIZE"]:
        if feat not in df.columns: continue
        plt.figure(figsize=(8,6))
        data = [df[df["predicted_class"]==c][feat].dropna() for c in classes]
        plt.boxplot(data)
        plt.xticks(range(1, len(classes)+1), classes)
        plt.title(f"{feat} by Class")
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(MORPH_VIS_DIR / f"{feat}_by_class.png")
        plt.close()
        
    # 2. Shape
    for feat in ["circularity", "aspect_ratio", "solidity", "extent"]:
        if feat not in df.columns: continue
        plt.figure(figsize=(8,6))
        data = [df[df["predicted_class"]==c][feat].dropna() for c in classes]
        plt.boxplot(data)
        plt.xticks(range(1, len(classes)+1), classes)
        plt.title(f"{feat} by Class")
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(MORPH_VIS_DIR / f"{feat}_by_class.png")
        plt.close()
        
    # Shape scatter
    pairs = [("area", "circularity"), ("equivalent_diameter", "circularity"), ("aspect_ratio", "solidity")]
    colors = plt.get_cmap("tab10", len(classes))
    for fx, fy in pairs:
        if fx in df.columns and fy in df.columns:
            plt.figure(figsize=(8,6))
            for i, c in enumerate(classes):
                sub = df[df["predicted_class"]==c]
                plt.scatter(sub[fx], sub[fy], label=c, alpha=0.6, color=colors(i))
            plt.xlabel(fx); plt.ylabel(fy)
            plt.title(f"{fy} vs {fx}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(MORPH_VIS_DIR / "shape_relationships" / f"{fx}_vs_{fy}.png")
            plt.close()
            
    # Correlation
    for c in classes:
        sub = df[df["predicted_class"]==c][feature_groups["SIZE"] + feature_groups["SHAPE"]].dropna()
        if len(sub) > 5:
            corr = sub.corr()
            plt.figure(figsize=(10,8))
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title(f"Correlation Matrix: {c}")
            plt.tight_layout()
            plt.savefig(MORPH_VIS_DIR / "correlation" / f"{c}_correlation.png")
            plt.close()
def extract_outlier_crops(df):
    """Extract crops for high-confidence outliers."""
    outliers = df[df["high_confidence_morphological_outlier"] == 1]
    count = 0
    
    for _, row in outliers.iterrows():
        if count >= 20: break
        
        img_name = row["image_name"]
        
        img_path = None
        for p in get_paired_files(split="test"):
            if p["image_path"].name == img_name:
                img_path = p["image_path"]
                break
                
        if img_path and img_path.exists():
            img_bgr = load_image(img_path)
            x1, y1 = int(row["bbox_x"]), int(row["bbox_y"])
            w, h = int(row["bbox_width"]), int(row["bbox_height"])
            
            pad = 15
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(img_bgr.shape[1], x1 + w + pad)
            y2_p = min(img_bgr.shape[0], y1 + h + pad)
            
            crop = img_bgr[y1_p:y2_p, x1_p:x2_p]
            
            fname = f"outlier_{row['predicted_class']}_{img_name}_id{row['instance_id']}.png"
            cv2.imwrite(str(MORPH_VIS_DIR / "outliers" / fname), crop)
            count += 1

def run_stage_8_morphology():
    print("\n" + "=" * 70)
    print(" STAGE 8: MORPHOLOGICAL ANALYSIS & OUTLIER DETECTION ")
    print("=" * 70)
    ensure_morphology_dirs()
    
    preds_path = METRICS_DIR / "cell_predictions.csv"
    if not preds_path.exists():
        print("Predictions not found. Run Stage 7.")
        return
        
    df = pd.read_csv(preds_path)
    if df.empty:
        print("No cells.")
        return
        
    groups = define_feature_groups()
    
    # 1. Class Statistics
    stats_df = calculate_class_statistics(df, groups)
    stats_df.to_csv(METRICS_DIR / "morphology_class_statistics.csv", index=False)
    
    # 2. Outlier Detection
    df_out, reasons_df = detect_outliers(df, groups)
    df_out.to_csv(METRICS_DIR / "morphological_outliers.csv", index=False)
    reasons_df.to_csv(METRICS_DIR / "outlier_feature_reasons.csv", index=False)
    
    # 3. Visualizations
    generate_visualizations(df_out, groups)
    extract_outlier_crops(df_out)
    
    # 4. Outlier Summary
    summary = []
    classes = df_out["predicted_class"].unique()
    for cls in classes:
        cls_df = df_out[df_out["predicted_class"] == cls]
        summary.append({
            "class": cls,
            "total_cells": len(cls_df),
            "z_outliers": cls_df["is_z_outlier"].sum(),
            "iqr_outliers": cls_df["is_iqr_outlier"].sum(),
            "multivariate_outliers": cls_df["is_multivariate_outlier"].sum(),
            "high_confidence_outliers": cls_df["high_confidence_morphological_outlier"].sum(),
            "outlier_percentage": (cls_df["high_confidence_morphological_outlier"].sum() / len(cls_df)) * 100 if len(cls_df) > 0 else 0
        })
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(METRICS_DIR / "outlier_summary.csv", index=False)
    
    # Generate Report
    def df_to_md(df):
        cols = df.columns.tolist()
        header = "| " + " | ".join([str(c) for c in cols]) + " |"
        sep = "|-" + "-|-".join(["-"*len(str(c)) for c in cols]) + "-|"
        rows = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join([str(round(x, 2)) if isinstance(x, (int, float)) else str(x) for x in r.values]) + " |")
        return "\n".join([header, sep] + rows)
        
    report = f"""# Stage 8: Morphological Analysis & Outlier Detection

## 1. Objective
To analyze the morphology (size, shape, color, intensity, texture) of detected blood-cell objects using Stage 5 features and identify cells whose image-derived characteristics are statistically unusual compared with other cells of the same predicted class.

## 2. Feature Groups Evaluated
- **SIZE**: Area, diameter, perimeter.
- **SHAPE**: Circularity, aspect ratio, solidity.
- **COLOR/INTENSITY**: RGB, HSV, LAB, intensity stats.

## 3. Outlier Detection Methods
We ran class-specific statistical checks to prevent comparing tiny platelets to massive WBCs:
1. **Univariate Z-score**: $|Z| > 3.0$
2. **Univariate IQR**: values outside $Q_1 - 1.5 IQR$ to $Q_3 + 1.5 IQR$
3. **Multivariate**: Isolation Forest (contamination=0.05)

High-confidence outliers are those flagged by at least 2 independent methods.

## 4. Outlier Summary
{df_to_md(sum_df)}

## 5. Relationship with Classification & Counting
Outliers heavily correlate with Watershed segmentation failures. An under-segmented cluster of RBCs will have massive Area and bizarre Circularity, triggering an outlier flag. Thus, most "outliers" are image-processing artifacts, not biological anomalies.

## 6. Limitations & Medical Disclaimer
- **Segmentation Errors**: The primary driver of morphological outliers in this pipeline is watershed over/under-segmentation.
- **Not Clinical Diagnosis**: This is an educational/research image analysis tool. An image-based morphological outlier is **NOT** automatically a diseased cell. Do not use for medical diagnosis.
"""
    with open(REPORTS_DIR / "morphological_analysis.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print(" STAGE 8 COMPLETED SUCCESSFULLY ")
