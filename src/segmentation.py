"""
Blood Cell Segmentation Module (Stage 3).
Provides classical thresholding (Otsu, Adaptive, Color-based, K-Means Clustering),
foreground polarity normalization, connected components analysis, and object-level GT box coverage validation.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Add parent directory to path to load config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTATION_CONFIG, CLASS_ID_TO_NAME
from .preprocessing import grayscale_conversion, bilateral_filter, clahe_enhancement
from .morphology import clean_segmentation_mask


def normalize_foreground_polarity(binary_mask):
    """
    Automatically detect and normalize binary mask polarity so that:
    - Cells (Foreground) = 255 (White)
    - Plasma (Background) = 0 (Black)
    
    Strategy:
    Examines the 4 outer borders of the image. Plasma background spans across image borders.
    If > 50% of border pixels are 255 (White), the mask is inverted so background becomes 0.
    
    Args:
        binary_mask (np.ndarray): 2D uint8 binary mask.
        
    Returns:
        np.ndarray: Polarity-normalized binary mask (Cells=255, Background=0).
    """
    h, w = binary_mask.shape[:2]

    # Sample border pixels (top, bottom, left, right edges)
    border_pixels = np.concatenate([
        binary_mask[0, :],         # Top edge
        binary_mask[h - 1, :],     # Bottom edge
        binary_mask[:, 0],         # Left edge
        binary_mask[:, w - 1],     # Right edge
    ])

    border_white_ratio = np.mean(border_pixels == 255)

    # If majority of border pixels are 255, background is currently white -> invert mask
    if border_white_ratio > 0.5:
        return 255 - binary_mask
    return binary_mask


def otsu_segmentation(image_gray):
    """
    Apply Global Otsu Thresholding.
    Calculates the optimal global threshold separating the bimodal intensity histogram.
    
    Args:
        image_gray (np.ndarray): 2D single-channel uint8 preprocessed image.
        
    Returns:
        tuple: (otsu_threshold_value, polarity_normalized_binary_mask)
    """
    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    # Otsu thresholding
    threshold_val, raw_mask = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Normalize polarity (Cells=255, Background=0)
    normalized_mask = normalize_foreground_polarity(raw_mask)
    return threshold_val, normalized_mask


def adaptive_segmentation(image_gray, method=None, block_size=None, c=None):
    """
    Apply Local Adaptive Thresholding.
    Computes local threshold for each pixel based on mean or Gaussian weighted sum of block_size x block_size neighborhood.
    Effective for images with non-uniform field illumination.
    
    Args:
        image_gray (np.ndarray): 2D uint8 grayscale image.
        method (str, optional): 'gaussian' or 'mean'.
        block_size (int, optional): Size of pixel neighborhood (odd integer).
        c (float, optional): Constant subtracted from mean/gaussian weighted sum.
        
    Returns:
        np.ndarray: Polarity-normalized binary mask.
    """
    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    if method is None:
        method = SEGMENTATION_CONFIG.get("adaptive_method", "gaussian")
    if block_size is None:
        block_size = SEGMENTATION_CONFIG.get("adaptive_block_size", 21)
    if c is None:
        c = SEGMENTATION_CONFIG.get("adaptive_c", 5.0)

    if block_size % 2 == 0:
        block_size += 1

    adaptive_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    raw_mask = cv2.adaptiveThreshold(
        image_gray, 255, adaptive_type, cv2.THRESH_BINARY, block_size, c
    )

    normalized_mask = normalize_foreground_polarity(raw_mask)
    return normalized_mask


def color_based_segmentation(image_bgr, color_space="HSV", lower_bound=None, upper_bound=None):
    """
    Apply Color-Based Segmentation using HSV or LAB color thresholding.
    
    In Romanowsky/Giemsa-stained blood smears:
    - HSV Saturation (S): Stained cells have higher saturation (S >= 30) than low-saturation plasma background.
    - LAB 'a' channel: High 'a' values separate pinkish RBC cytoplasm and purple WBC nuclei from background.
    
    Args:
        image_bgr (np.ndarray): Input BGR image array.
        color_space (str): Color space name ('HSV' or 'LAB').
        lower_bound (tuple/int, optional): Lower threshold bound.
        upper_bound (tuple/int, optional): Upper threshold bound.
        
    Returns:
        np.ndarray: Polarity-normalized binary mask.
    """
    if color_space.upper() == "HSV":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        lower_s = lower_bound if lower_bound is not None else SEGMENTATION_CONFIG.get("hsv_s_lower", 30)
        upper_s = upper_bound if upper_bound is not None else SEGMENTATION_CONFIG.get("hsv_s_upper", 255)

        raw_mask = cv2.inRange(s_channel, lower_s, upper_s)

    elif color_space.upper() == "LAB":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]

        # Cells have higher 'a' component (towards magenta/red)
        threshold_val, raw_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unsupported color space for segmentation: {color_space}")

    normalized_mask = normalize_foreground_polarity(raw_mask)
    return normalized_mask


def kmeans_segmentation(image_bgr, k=None):
    """
    Apply Classical K-Means Color/Intensity Clustering.
    Clusters pixels into K color/intensity groups and automatically assigns the background cluster based on border pixel frequency.
    
    Args:
        image_bgr (np.ndarray): Input BGR image array.
        k (int, optional): Number of clusters. Defaults to config (k=3).
        
    Returns:
        np.ndarray: Polarity-normalized binary mask (Cells=255, Background=0).
    """
    if k is None:
        k = SEGMENTATION_CONFIG.get("kmeans_k", 3)

    h, w, c = image_bgr.shape
    pixel_values = image_bgr.reshape((-1, 3)).astype(np.float32)

    # K-Means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # OpenCV uses a process-global RNG. Reset it so a given image/configuration
    # produces repeatable masks and comparable experiment rows.
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    labels = labels.reshape((h, w))

    # Identify background cluster index (cluster most prevalent along border pixels)
    border_pixels = np.concatenate([
        labels[0, :], labels[h - 1, :], labels[:, 0], labels[:, w - 1]
    ])
    bg_cluster = int(np.bincount(border_pixels).argmax())

    # Foreground is all non-background clusters
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    raw_mask[labels != bg_cluster] = 255

    return raw_mask


def analyze_connected_components(cleaned_mask):
    """
    Perform 8-connectivity Connected Component Analysis on a cleaned binary mask.
    Calculates component area, bounding box coordinates (x, y, w, h), and centroids.
    
    Args:
        cleaned_mask (np.ndarray): 2D uint8 cleaned binary mask (0 or 255).
        
    Returns:
        tuple: (num_components, labeled_mask, components_dataframe)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

    rows = []
    for label_idx in range(1, num_labels):  # Label 0 is background
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
    # Total count of cell components excluding background
    num_components = num_labels - 1
    return num_components, labels, df


def validate_object_level_coverage(annotations, binary_mask, coverage_threshold=0.10):
    """
    Perform Object-Level Validation comparing segmentation mask foreground pixels against Ground-Truth YOLO boxes.
    
    IMPORTANT NOTE:
    TXL-PBC provides rectangular bounding box annotations, not pixel-level segmentation masks.
    This function computes:
    1. `foreground_ratio`: Proportion of foreground pixels (255) inside each GT box.
    2. `gt_box_coverage`: Percentage of GT boxes containing >= `coverage_threshold` (e.g. 10%) foreground pixels.
    
    This measures object-level presence/detection validation, NOT pixel IoU.
    
    Args:
        annotations (list of dict): Parsed ground-truth YOLO annotations.
        binary_mask (np.ndarray): 2D uint8 binary mask (0 or 255).
        coverage_threshold (float): Minimum foreground ratio required to count box as covered.
        
    Returns:
        dict: Validation summary metrics and per-box details.
    """
    if not annotations:
        return {
            "total_gt_boxes": 0,
            "covered_gt_boxes": 0,
            "coverage_rate_pct": 0.0,
            "mean_fg_ratio_inside_box": 0.0,
            "per_box_results": [],
        }

    h_img, w_img = binary_mask.shape[:2]
    per_box_results = []
    fg_ratios = []
    covered_count = 0

    for idx, ann in enumerate(annotations, 1):
        x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
        cname = ann["class_name"]

        # Ensure coordinates are within valid image dimensions
        x1_cl = max(0, min(x1, w_img - 1))
        y1_cl = max(0, min(y1, h_img - 1))
        x2_cl = max(x1_cl + 1, min(x2, w_img))
        y2_cl = max(y1_cl + 1, min(y2, h_img))

        crop = binary_mask[y1_cl:y2_cl, x1_cl:x2_cl]
        box_area = crop.size

        if box_area > 0:
            fg_pixels = np.count_nonzero(crop == 255)
            fg_ratio = fg_pixels / float(box_area)
        else:
            fg_pixels = 0
            fg_ratio = 0.0

        is_covered = fg_ratio >= coverage_threshold
        if is_covered:
            covered_count += 1

        fg_ratios.append(fg_ratio)
        per_box_results.append({
            "box_id": idx,
            "class_name": cname,
            "bbox_pixel": [x1_cl, y1_cl, x2_cl, y2_cl],
            "fg_pixels": fg_pixels,
            "box_area": box_area,
            "fg_ratio": round(fg_ratio, 4),
            "is_covered": is_covered,
        })

    coverage_rate_pct = (covered_count / float(len(annotations))) * 100.0 if annotations else 0.0
    mean_fg_ratio = float(np.mean(fg_ratios)) if fg_ratios else 0.0

    return {
        "total_gt_boxes": len(annotations),
        "covered_gt_boxes": covered_count,
        "coverage_rate_pct": round(coverage_rate_pct, 2),
        "mean_fg_ratio_inside_box": round(mean_fg_ratio, 4),
        "per_box_results": per_box_results,
    }


def segment_cells(image_bgr, method=None, channel=None, options=None):
    """
    Main entry point for Stage 3 Cell Segmentation.
    Loads input, extracts specified preprocessed channel, applies segmentation thresholding,
    normalizes polarity, and performs morphological post-processing cleanup.
    
    Args:
        image_bgr (np.ndarray): Input BGR image array.
        method (str, optional): Segmentation method ('otsu', 'adaptive', 'color_range', 'kmeans').
        channel (str, optional): Channel strategy ('green', 'clahe_green', 'grayscale', 'hsv_s', 'lab_l').
        options (dict, optional): Custom override options.
        
    Returns:
        dict: Complete segmentation result containing preprocessed image, raw mask, cleaned mask,
              connected components DataFrame, and Otsu threshold (if applicable).
    """
    if options is None:
        options = {}

    if method is None:
        method = SEGMENTATION_CONFIG.get("segmentation_method", "otsu")
    if channel is None:
        channel = SEGMENTATION_CONFIG.get("input_channel", "green")

    # Step 1: Preprocessed channel extraction
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
    elif channel == "lab_a":
        preprocessed_img = grayscale_conversion(image_bgr, method="lab_a")
    else:
        preprocessed_img = grayscale_conversion(image_bgr, method="standard")

    # Step 2: Apply Segmentation Method
    threshold_val = None

    if method == "triangle":
        # Triangle thresholding requires a single-channel 8-bit image
        if len(preprocessed_img.shape) > 2:
            preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        thresh_val, raw_mask = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        
    elif method == "otsu":
        threshold_val, raw_mask = otsu_segmentation(preprocessed_img)
    elif method == "adaptive":
        raw_mask = adaptive_segmentation(preprocessed_img)
    elif method == "color_range":
        raw_mask = color_based_segmentation(image_bgr, color_space="HSV")
    elif method == "kmeans":
        raw_mask = kmeans_segmentation(image_bgr)
    else:
        threshold_val, raw_mask = otsu_segmentation(preprocessed_img)

    # Step 3: Morphological Post-Processing Cleanup
    cleaned_mask = clean_segmentation_mask(raw_mask, options)

    # Step 4: Connected Component Analysis
    num_components, labeled_mask, comp_df = analyze_connected_components(cleaned_mask)

    return {
        "preprocessed_img": preprocessed_img,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "labeled_mask": labeled_mask,
        "num_components": num_components,
        "components_df": comp_df,
        "otsu_threshold": threshold_val,
        "method": method,
        "channel": channel,
    }
