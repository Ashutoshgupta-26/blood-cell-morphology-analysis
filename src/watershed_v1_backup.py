"""
Watershed Segmentation Module (Stage 4).
Implements Distance Transform and Watershed algorithm for separating touching and overlapping cells.
"""

import cv2
import numpy as np
import pandas as pd
from config import WATERSHED_CONFIG

def compute_distance_transform(binary_mask):
    """Compute the Euclidean distance transform of a binary mask."""
    mask_uint8 = np.zeros_like(binary_mask, dtype=np.uint8)
    mask_uint8[binary_mask > 0] = 255
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 3)
    return dist_transform

def generate_foreground_markers(dist_transform, binary_mask, threshold_ratio):
    """
    Generate foreground markers by thresholding the distance transform
    PER component to ensure small cells (like platelets) are not lost.
    """
    sure_fg = np.zeros_like(dist_transform, dtype=np.uint8)
    
    # Process each component from the Stage 3 mask separately
    num_comp, labels = cv2.connectedComponents(binary_mask)
    for i in range(1, num_comp + 1):
        comp_mask = (labels == i)
        comp_dist = np.zeros_like(dist_transform)
        comp_dist[comp_mask] = dist_transform[comp_mask]
        
        max_val = comp_dist.max()
        if max_val > 0:
            # Threshold locally for this component
            _, fg = cv2.threshold(comp_dist, threshold_ratio * max_val, 255, 0)
            sure_fg = cv2.bitwise_or(sure_fg, np.uint8(fg))
            
    # Run connected components on the combined foreground markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    return sure_fg, markers, num_labels

def generate_background_markers(binary_mask, kernel_size=(3, 3), iterations=3):
    """
    Generate sure background by dilating the original binary mask.
    """
    mask_uint8 = np.zeros_like(binary_mask, dtype=np.uint8)
    mask_uint8[binary_mask > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    sure_bg = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    return sure_bg

def apply_watershed(image_bgr, sure_bg, sure_fg, markers):
    """
    Apply OpenCV's watershed algorithm.
    """
    # Create the unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers_copy = markers.copy() + 1
    
    # Mark the region of unknown with zero
    markers_copy[unknown == 255] = 0
    
    # Apply watershed
    # Watershed modifies markers in-place. Boundary pixels get -1.
    markers_watershed = cv2.watershed(image_bgr, markers_copy)
    return markers_watershed

def separate_touching_cells(binary_mask, image_bgr, threshold_ratio=0.3):
    """
    Apply Watershed algorithm to separate touching/overlapping cell clusters.
    Uses configurable distance transform thresholding.
    """
    dist_transform = compute_distance_transform(binary_mask)
    sure_fg, markers, num_labels = generate_foreground_markers(dist_transform, binary_mask, threshold_ratio)
    sure_bg = generate_background_markers(binary_mask)
    
    # Ensure image is 8-bit 3-channel for cv2.watershed
    img_bgr = image_bgr.copy()
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
    labels = apply_watershed(img_bgr, sure_bg, sure_fg, markers)
    
    # Re-map labels so background is 0, cells are 1..N
    # In OpenCV watershed, 1 was sure_bg, >1 are cells, -1 are boundaries.
    # We want 0 for bg, positive integers for cells.
    
    min_area = WATERSHED_CONFIG.get("min_cell_area", 100)
    
    final_labels = np.zeros_like(labels, dtype=np.int32)
    current_label = 1
    for label_id in range(2, num_labels + 1):
        mask = (labels == label_id)
        if np.count_nonzero(mask) >= min_area:
            final_labels[mask] = current_label
            current_label += 1
            
    # Assign boundaries (-1) to background (0) or keep them. We'll assign to background to fully separate cells.
    
    return {
        "labels": final_labels,
        "dist_transform": dist_transform,
        "markers": markers,  # Original foreground markers
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "raw_watershed_labels": labels
    }

def extract_instance_contours(labels):
    """Extract contours for each unique instance label."""
    contours_dict = {}
    num_cells = labels.max()
    for i in range(1, num_cells + 1):
        cell_mask = np.uint8(labels == i) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Assuming one main contour per cell
            main_contour = max(contours, key=cv2.contourArea)
            contours_dict[i] = main_contour
    return contours_dict

def compute_instance_statistics(labels, image_name):
    """
    Compute area, bounding box, centroid, and perimeter for each instance.
    """
    stats_list = []
    num_cells = labels.max()
    for i in range(1, num_cells + 1):
        cell_mask = np.uint8(labels == i) * 255
        area = cv2.countNonZero(cell_mask)
        if area == 0:
            continue
            
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            cX, cY = x + w/2, y + h/2
            
        stats_list.append({
            "image_name": image_name,
            "instance_id": i,
            "area": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": w,
            "bbox_height": h,
            "centroid_x": cX,
            "centroid_y": cY,
            "perimeter": perimeter
        })
    return pd.DataFrame(stats_list)

def analyze_gt_overlap(labels, annotations, image_name):
    """
    Check overlap of watershed instances with ground-truth YOLO bounding boxes.
    """
    gt_stats = []
    for ann in annotations:
        gx, gy = int(ann["x1"]), int(ann["y1"])
        gw, gh = int(ann["width_px"]), int(ann["height_px"])
        
        # Create a mask for the GT box
        gt_mask = np.zeros_like(labels, dtype=np.uint8)
        gt_mask[gy:gy+gh, gx:gx+gw] = 1
        
        # Find which labels overlap this box
        overlap = labels[gt_mask == 1]
        unique_labels = np.unique(overlap)
        unique_labels = unique_labels[unique_labels > 0] # Exclude background
        
        best_overlap_ratio = 0.0
        for l_id in unique_labels:
            inst_mask = (labels == l_id)
            inst_area = np.count_nonzero(inst_mask)
            intersect = np.count_nonzero(inst_mask & (gt_mask == 1))
            if inst_area > 0:
                ratio = intersect / inst_area
                if ratio > best_overlap_ratio:
                    best_overlap_ratio = ratio
                    
        gt_stats.append({
            "image_name": image_name,
            "gt_class": ann["class_name"],
            "gt_x": gx,
            "gt_y": gy,
            "gt_width": gw,
            "gt_height": gh,
            "num_watershed_instances_overlapping": len(unique_labels),
            "best_overlap_ratio": round(best_overlap_ratio, 4)
        })
    return pd.DataFrame(gt_stats)
