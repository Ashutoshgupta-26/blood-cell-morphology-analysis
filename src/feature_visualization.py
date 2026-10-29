"""
Feature visualization functions (Stage 5).
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_feature_values_on_cells(image_bgr, features_df, save_path):
    """Draw instance ID, Area, and Circularity on the image."""
    annotated = image_bgr.copy()
    for _, row in features_df.iterrows():
        cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
        text = f"ID:{int(row['instance_id'])}\nA:{int(row['area'])}\nC:{row['circularity']:.2f}"
        
        y_offset = cy - 10
        for line in text.split('\n'):
            cv2.putText(annotated, line, (cx - 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 12
            
    cv2.imwrite(str(save_path), annotated)

def visualize_gt_matching(image_bgr, matches, labels, save_path):
    """Draw GT boxes and indicate match status."""
    annotated = image_bgr.copy()
    
    # Draw instance boundaries
    contour_img = np.zeros_like(image_bgr)
    num_cells = labels.max()
    for i in range(1, num_cells + 1):
        mask = np.uint8(labels == i) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 255, 0), 1)
        
        # Draw ID
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(annotated, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
    # Draw GT Boxes
    for match in matches:
        x1, y1 = match["gt_x1"], match["gt_y1"]
        x2, y2 = x1 + match["gt_w"], y1 + match["gt_h"]
        status = match["match_status"]
        
        color = (0, 255, 0) if status == "matched" else (0, 0, 255) if status == "unmatched" else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        label = f"{match['gt_class']} ({status})"
        if status == "matched":
            label += f" -> ID:{match['best_instance_id']}"
            
        cv2.putText(annotated, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
    cv2.imwrite(str(save_path), annotated)

def plot_feature_distributions(features_df, output_dir):
    """Plot histograms for key features."""
    features_to_plot = ["area", "equivalent_diameter", "circularity", "aspect_ratio", "solidity", "mean_intensity"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for feat in features_to_plot:
        if feat in features_df.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(features_df[feat].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f"Distribution of {feat}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{feat}_distribution.png")
            plt.close()

def plot_feature_correlation(features_df, save_path):
    """Plot correlation matrix."""
    cols = [
        "area", "perimeter", "equivalent_diameter", "circularity", "aspect_ratio", 
        "extent", "solidity", "mean_intensity", "std_intensity", 
        "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v", 
        "mean_l", "mean_a", "mean_lab_b", "mean_gradient", "std_gradient"
    ]
    # Filter available
    cols = [c for c in cols if c in features_df.columns]
    if not cols: return
    
    corr = features_df[cols].corr()
    
    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return corr

def plot_feature_separability(gt_features_df, output_dir):
    """Plot exploratory separability plots."""
    if gt_features_df.empty or "gt_class" not in gt_features_df.columns:
        return
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = [
        ("area", "circularity"),
        ("area", "mean_intensity"),
        ("area", "mean_s"),
        ("equivalent_diameter", "circularity")
    ]
    
    classes = gt_features_df["gt_class"].unique()
    colors = plt.get_cmap("tab10", len(classes))
    
    for x_feat, y_feat in pairs:
        if x_feat in gt_features_df.columns and y_feat in gt_features_df.columns:
            plt.figure(figsize=(8, 6))
            for i, c in enumerate(classes):
                subset = gt_features_df[gt_features_df["gt_class"] == c]
                plt.scatter(subset[x_feat], subset[y_feat], label=c, color=colors(i), alpha=0.7)
            plt.xlabel(x_feat)
            plt.ylabel(y_feat)
            plt.legend()
            plt.title(f"{y_feat} vs {x_feat} by Class")
            plt.tight_layout()
            plt.savefig(output_dir / f"{x_feat}_vs_{y_feat}.png")
            plt.close()
            
    # Boxplots via matplotlib
    for feat in ["area", "circularity", "mean_s", "mean_intensity"]:
        if feat in gt_features_df.columns:
            plt.figure(figsize=(8, 6))
            data = [gt_features_df[gt_features_df["gt_class"] == c][feat].dropna() for c in classes]
            plt.boxplot(data)
            plt.xticks(range(1, len(classes) + 1), classes)
            plt.title(f"{feat} by Class")
            plt.tight_layout()
            plt.savefig(output_dir / f"boxplot_{feat}.png")
            plt.close()
