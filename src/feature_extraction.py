"""
Feature Extraction Module (Stage 5).
Extracts geometric, morphological, color, intensity, and texture features
from individual cell instances produced by watershed segmentation.
"""

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from .preprocessing import convert_color_spaces

def calculate_texture_gradient(img_gray):
    """Calculate gradient magnitude using Sobel."""
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    return grad_mag

def extract_features_for_image(image_bgr, labels, image_name):
    """
    Extract comprehensive features for all cell instances in an image.
    Calculates image-level color conversions once for efficiency.
    """
    if labels.max() == 0:
        return []

    # Pre-calculate full image representations once
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_mag = calculate_texture_gradient(img_gray)
    
    height, width = image_bgr.shape[:2]

    # Use skimage regionprops for geometric features on the label image
    props = regionprops(labels)
    
    features_list = []
    
    for prop in props:
        instance_id = prop.label
        
        # 1. Geometry & Shape Features
        area = prop.area
        perimeter = prop.perimeter
        y_min, x_min, y_max, x_max = prop.bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        centroid_y, centroid_x = prop.centroid
        
        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
        extent = prop.extent  # area / (bbox_w * bbox_h)
        
        equivalent_diameter = prop.equivalent_diameter_area
        
        # Circularity (4*pi*area / perimeter^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        convex_area = prop.area_convex
        solidity = prop.solidity
        orientation = prop.orientation
        
        # Enclosing circle radius via OpenCV
        cell_mask = np.uint8(labels == instance_id)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            _, radius = cv2.minEnclosingCircle(main_contour)
        else:
            radius = 0

        # Quality flags
        touches_border = (x_min == 0) or (y_min == 0) or (x_max == width) or (y_max == height)
        very_small = area < 50
        very_large = area > 10000
        
        # Mask boolean array for pixel extraction
        mask_bool = (labels == instance_id)
        
        # 2. Color Features
        r_vals = img_rgb[mask_bool, 0]
        g_vals = img_rgb[mask_bool, 1]
        b_vals = img_rgb[mask_bool, 2]
        
        h_vals = img_hsv[mask_bool, 0]
        s_vals = img_hsv[mask_bool, 1]
        v_vals = img_hsv[mask_bool, 2]
        
        l_vals = img_lab[mask_bool, 0]
        a_vals = img_lab[mask_bool, 1]
        b_lab_vals = img_lab[mask_bool, 2]
        
        # 3. Intensity Features
        gray_vals = img_gray[mask_bool]
        
        # 4. Texture Features (Gradient)
        grad_vals = grad_mag[mask_bool]
        
        row = {
            "image_name": image_name,
            "image_width": width,
            "image_height": height,
            "instance_id": instance_id,
            
            # Geometry
            "area": area,
            "perimeter": perimeter,
            "bbox_x": x_min,
            "bbox_y": y_min,
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "normalized_centroid_x": centroid_x / width,
            "normalized_centroid_y": centroid_y / height,
            
            # Shape
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "equivalent_diameter": equivalent_diameter,
            "circularity": circularity,
            "convex_hull_area": convex_area,
            "solidity": solidity,
            "orientation": orientation,
            "enclosing_circle_radius": radius,
            
            # Color RGB
            "mean_r": r_vals.mean() if len(r_vals)>0 else 0,
            "mean_g": g_vals.mean() if len(g_vals)>0 else 0,
            "mean_b": b_vals.mean() if len(b_vals)>0 else 0,
            "std_r": r_vals.std() if len(r_vals)>0 else 0,
            "std_g": g_vals.std() if len(g_vals)>0 else 0,
            "std_b": b_vals.std() if len(b_vals)>0 else 0,
            
            # Color HSV
            "mean_h": h_vals.mean() if len(h_vals)>0 else 0,
            "mean_s": s_vals.mean() if len(s_vals)>0 else 0,
            "mean_v": v_vals.mean() if len(v_vals)>0 else 0,
            "std_h": h_vals.std() if len(h_vals)>0 else 0,
            "std_s": s_vals.std() if len(s_vals)>0 else 0,
            "std_v": v_vals.std() if len(v_vals)>0 else 0,
            
            # Color LAB
            "mean_l": l_vals.mean() if len(l_vals)>0 else 0,
            "mean_a": a_vals.mean() if len(a_vals)>0 else 0,
            "mean_lab_b": b_lab_vals.mean() if len(b_lab_vals)>0 else 0,
            "std_l": l_vals.std() if len(l_vals)>0 else 0,
            "std_a": a_vals.std() if len(a_vals)>0 else 0,
            "std_lab_b": b_lab_vals.std() if len(b_lab_vals)>0 else 0,
            
            # Intensity
            "mean_intensity": gray_vals.mean() if len(gray_vals)>0 else 0,
            "median_intensity": np.median(gray_vals) if len(gray_vals)>0 else 0,
            "std_intensity": gray_vals.std() if len(gray_vals)>0 else 0,
            "min_intensity": gray_vals.min() if len(gray_vals)>0 else 0,
            "max_intensity": gray_vals.max() if len(gray_vals)>0 else 0,
            "intensity_p25": np.percentile(gray_vals, 25) if len(gray_vals)>0 else 0,
            "intensity_p75": np.percentile(gray_vals, 75) if len(gray_vals)>0 else 0,
            
            # Texture
            "mean_gradient": grad_vals.mean() if len(grad_vals)>0 else 0,
            "std_gradient": grad_vals.std() if len(grad_vals)>0 else 0,
            
            # Quality Flags
            "touches_border": int(touches_border),
            "very_small": int(very_small),
            "very_large": int(very_large)
        }
        features_list.append(row)
        
    return features_list

def associate_ground_truth(features_df, labels, annotations, image_name, iou_threshold=0.3):
    """
    Match watershed instances with GT bounding boxes based on Box IoU.
    Returns list of match dictionaries.
    """
    matches = []
    
    if features_df.empty or not annotations:
        return matches
        
    for ann in annotations:
        gt_x1, gt_y1 = int(ann["x1"]), int(ann["y1"])
        gt_x2, gt_y2 = int(ann["x2"]), int(ann["y2"])
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        gt_area = gt_w * gt_h
        gt_class = ann["class_name"]
        gt_class_id = ann["class_id"]
        
        best_iou = 0.0
        best_instance = -1
        overlapping_instances = []
        
        for _, row in features_df.iterrows():
            inst_id = row["instance_id"]
            inst_x1 = row["bbox_x"]
            inst_y1 = row["bbox_y"]
            inst_w = row["bbox_width"]
            inst_h = row["bbox_height"]
            inst_x2 = inst_x1 + inst_w
            inst_y2 = inst_y1 + inst_h
            inst_area = inst_w * inst_h
            
            # Calculate Box Intersection
            ix1 = max(gt_x1, inst_x1)
            iy1 = max(gt_y1, inst_y1)
            ix2 = min(gt_x2, inst_x2)
            iy2 = min(gt_y2, inst_y2)
            
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih
            
            if inter_area > 0:
                overlapping_instances.append(inst_id)
                iou = inter_area / float(gt_area + inst_area - inter_area)
                if iou > best_iou:
                    best_iou = iou
                    best_instance = inst_id
                    
        match_status = "unmatched"
        if best_iou >= iou_threshold:
            if len(overlapping_instances) > 1 and best_iou < 0.6:
                match_status = "ambiguous"
            else:
                match_status = "matched"
                
        matches.append({
            "image_name": image_name,
            "gt_class": gt_class,
            "gt_class_id": gt_class_id,
            "gt_x1": gt_x1, "gt_y1": gt_y1,
            "gt_w": gt_w, "gt_h": gt_h,
            "best_instance_id": best_instance,
            "match_iou": round(best_iou, 4),
            "match_status": match_status,
            "num_overlapping_instances": len(overlapping_instances)
        })
        
    return matches
