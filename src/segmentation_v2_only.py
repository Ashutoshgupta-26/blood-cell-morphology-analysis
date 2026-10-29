"""
Cell Segmentation Module V2 (Stage 3)
Replaces the legacy Otsu-based approach with Triangle Thresholding and 
conservative morphological reconstruction to prevent crescent fragments.
"""

import cv2
import numpy as np
import pandas as pd
from config import SEGMENTATION_CONFIG
from src.preprocessing import grayscale_conversion, bilateral_filter, clahe_enhancement
from src.morphology import clean_segmentation_mask

def triangle_segmentation(preprocessed_img):
    """
    Applies Triangle Thresholding to find the optimal threshold for RBCs.
    Unlike Otsu (which struggles with the multi-modal background+rim+pallor distribution),
    Triangle thresholding correctly models the long tail of the cell intensities against
    the massive background peak, ensuring the cell rims remain thick enough to close.
    """
    # Triangle thresholding requires a single-channel 8-bit image
    if len(preprocessed_img.shape) > 2:
        preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        
    thresh_val, raw_mask = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    return thresh_val, raw_mask

def analyze_connected_components(cleaned_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

    rows = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        cx = float(centroids[label_idx][0])
        cy = float(centroids[label_idx][1])

        rows.append({
            "component_id": label_idx,
            "area": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "centroid_x": round(cx, 2),
            "centroid_y": round(cy, 2),
        })

    df = pd.DataFrame(rows)
    num_components = num_labels - 1
    return num_components, labels, df

def segment_cells(image_bgr, method=None, channel=None, options=None):
    """
    Main entry point for Stage 3 Cell Segmentation V2.
    """
    if options is None: options = {}
    if method is None: method = SEGMENTATION_CONFIG.get("segmentation_method", "triangle")
    if channel is None: channel = SEGMENTATION_CONFIG.get("input_channel", "green")
    
    # 1. Channel Extraction
    if channel == "clahe_green":
        green = grayscale_conversion(image_bgr, method="green")
        filtered = bilateral_filter(green)
        preprocessed_img = clahe_enhancement(filtered)
    elif channel == "green":
        green = grayscale_conversion(image_bgr, method="green")
        preprocessed_img = bilateral_filter(green)
    elif channel == "hsv_s":
        preprocessed_img = grayscale_conversion(image_bgr, method="hsv_s")
    elif channel == "lab_l":
        preprocessed_img = grayscale_conversion(image_bgr, method="lab_l")
    else:
        preprocessed_img = grayscale_conversion(image_bgr, method="standard")
        
    # 2. Thresholding
    if method == "triangle":
        thresh_val, raw_mask = triangle_segmentation(preprocessed_img)
    elif method == "otsu":
        from src.segmentation_legacy import otsu_segmentation
        thresh_val, raw_mask = otsu_segmentation(preprocessed_img)
    else:
        # Fallback to triangle
        thresh_val, raw_mask = triangle_segmentation(preprocessed_img)
        
    # 3. Morphological Reconstruction (Closing + Hole Filling)
    cleaned_mask = clean_segmentation_mask(raw_mask, options)
    
    # 4. Component Analysis
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        components.append({
            "component_id": i,
            "area": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "centroid_x": cx,
            "centroid_y": cy
        })
        
    df_components = pd.DataFrame(components) if components else pd.DataFrame(columns=["component_id", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "centroid_x", "centroid_y"])
    
    return {
        "preprocessed_img": preprocessed_img,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "components_df": df_components,
        "threshold_value": thresh_val,
        "labels": labels_im
    }
