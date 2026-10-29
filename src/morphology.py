"""
Morphological Operations and Post-Processing Module (Stage 3).
Provides operations for noise suppression, hole filling, small component removal,
and border object handling on binary cell segmentation masks.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path to load config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTATION_CONFIG


def get_structuring_element(kernel_size=(3, 3), shape=cv2.MORPH_ELLIPSE):
    """
    Generate an OpenCV structuring element (kernel) for morphological operations.
    Defaults to ELLIPSE shape which provides isotropic circular symmetry suitable for blood cells.
    
    Args:
        kernel_size (tuple): Width and height of kernel (must be odd numbers).
        shape (int): OpenCV morph shape (MORPH_ELLIPSE, MORPH_RECT, MORPH_CROSS).
        
    Returns:
        np.ndarray: Structuring element array.
    """
    k_w = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_h = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    return cv2.getStructuringElement(shape, (k_w, k_h))


def morphological_opening(mask, kernel_size=None):
    """
    Apply Morphological Opening (Erosion followed by Dilation).
    
    Purpose:
    Removes small isolated noise points, thin background bridges, and spicules without altering the
    overall area and geometry of larger blood cells.
    
    Args:
        mask (np.ndarray): 2D uint8 binary mask (0 or 255).
        kernel_size (tuple, optional): Kernel dimensions.
        
    Returns:
        np.ndarray: Cleaned binary mask.
    """
    if kernel_size is None:
        kernel_size = SEGMENTATION_CONFIG.get("morph_kernel_size", (3, 3))
    kernel = get_structuring_element(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def morphological_closing(mask, kernel_size=None):
    """
    Apply Morphological Closing (Dilation followed by Erosion).
    
    Purpose:
    Fills small internal gaps, cracks, and indentations along cell boundaries, bridging minor discontinuities.
    
    Args:
        mask (np.ndarray): 2D uint8 binary mask (0 or 255).
        kernel_size (tuple, optional): Kernel dimensions.
        
    Returns:
        np.ndarray: Cleaned binary mask.
    """
    if kernel_size is None:
        kernel_size = SEGMENTATION_CONFIG.get("morph_kernel_size", (3, 3))
    kernel = get_structuring_element(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def morphological_erosion(mask, kernel_size=None):
    """
    Apply Morphological Erosion to shrink cell foreground boundaries.
    """
    if kernel_size is None:
        kernel_size = SEGMENTATION_CONFIG.get("morph_kernel_size", (3, 3))
    kernel = get_structuring_element(kernel_size)
    return cv2.erode(mask, kernel, iterations=1)


def morphological_dilation(mask, kernel_size=None):
    """
    Apply Morphological Dilation to expand cell foreground boundaries.
    """
    if kernel_size is None:
        kernel_size = SEGMENTATION_CONFIG.get("morph_kernel_size", (3, 3))
    kernel = get_structuring_element(kernel_size)
    return cv2.dilate(mask, kernel, iterations=1)


def morphological_gradient(mask, kernel_size=None):
    """
    Calculate Morphological Gradient (Dilation minus Erosion).
    Yields thin 1-pixel boundary outlines of cell regions.
    """
    if kernel_size is None:
        kernel_size = SEGMENTATION_CONFIG.get("morph_kernel_size", (3, 3))
    kernel = get_structuring_element(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def fill_holes(mask, max_hole_area=5000):
    """
    Fill interior holes/cavities inside cell regions.
    Uses contour detection to only fill holes smaller than max_hole_area,
    preventing catastrophic filling of the entire image due to vignetting rings.
    
    Args:
        mask (np.ndarray): 2D uint8 binary mask (0 or 255).
        max_hole_area (int): Maximum area of a hole to fill.
        
    Returns:
        np.ndarray: Binary mask with solid interior cells.
    """
    if mask.size == 0 or not np.any(mask):
        return mask.copy()
    if np.all(mask):
        return mask.copy()

    # Find contours using RETR_CCOMP to identify holes inside foreground
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    result = mask.copy()
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # hierarchy[0][i] = [Next, Previous, First_Child, Parent]
            parent_idx = hierarchy[0][i][3]
            
            # If parent_idx != -1, this contour is a hole INSIDE a foreground object
            if parent_idx != -1:
                area = cv2.contourArea(contour)
                if area < max_hole_area:
                    cv2.drawContours(result, [contour], -1, 255, thickness=cv2.FILLED)
                    
    return result


def remove_small_objects(mask, min_area=None):
    """
    Filter out small connected components below a specified minimum area threshold.
    
    Important Note:
    Platelets in blood smears are small cells (typically 20-100 pixels in area depending on resolution).
    Setting min_area=20 successfully removes tiny 1-5 pixel noise artifacts without removing platelets.
    
    Args:
        mask (np.ndarray): 2D uint8 binary mask (0 or 255).
        min_area (int, optional): Minimum area in pixels. Defaults to config.
        
    Returns:
        np.ndarray: Filtered binary mask.
    """
    if min_area is None:
        min_area = SEGMENTATION_CONFIG.get("min_component_area", 20)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    # Label 0 is background; iterate through foreground labels 1..num_labels-1
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 255

    return cleaned


def remove_border_objects(mask):
    """
    Identify and remove connected components touching any of the 4 outer image borders.
    
    Note on Usage:
    Made optional (default False) because cells touching image borders in blood smears are valid real cells.
    Removing them outright would artificially reduce total cell count and ground-truth coverage.
    
    Args:
        mask (np.ndarray): 2D uint8 binary mask (0 or 255).
        
    Returns:
        np.ndarray: Binary mask with border-touching objects removed.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = mask.copy()

    h, w = mask.shape[:2]

    for label_idx in range(1, num_labels):
        # Check if component pixels touch image boundaries
        component_mask = (labels == label_idx)
        touches_top = np.any(component_mask[0, :])
        touches_bottom = np.any(component_mask[h - 1, :])
        touches_left = np.any(component_mask[:, 0])
        touches_right = np.any(component_mask[:, w - 1])

        if touches_top or touches_bottom or touches_left or touches_right:
            cleaned[component_mask] = 0

    return cleaned


def clean_segmentation_mask(raw_mask, options=None):
    """
    Execute full morphological cleaning pipeline on a raw binary segmentation mask.
    
    Pipeline Steps:
    1. Fill interior holes inside cells (`fill_holes`).
    2. Morphological Opening (`morphological_opening`) to eliminate thin background noise.
    3. Morphological Closing (`morphological_closing`) to smooth cell contours.
    4. Remove small noise components below `min_component_area` (`remove_small_objects`).
    5. Optionally remove border objects if enabled.
    
    Args:
        raw_mask (np.ndarray): Raw 2D uint8 binary mask (0 or 255).
        options (dict, optional): Custom post-processing overrides.
        
    Returns:
        np.ndarray: Fully cleaned 2D uint8 binary mask (0 or 255).
    """
    if options is None:
        options = {}

    kernel_size = options.get("kernel_size", SEGMENTATION_CONFIG.get("morph_kernel_size", (3, 3)))
    min_area = options.get("min_area", SEGMENTATION_CONFIG.get("min_component_area", 20))
    remove_border = options.get("remove_border", SEGMENTATION_CONFIG.get("remove_border_objects", False))
    close_before_fill = options.get("close_before_fill", True)

    if close_before_fill:
        # Close remaining gaps in cell rims before hole filling.
        # 7x7 is optimal for Triangle threshold: bridges pallor gaps without merging touching cells.
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_for_fill = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, bridge_kernel)
        filled = fill_holes(closed_for_fill)
    else:
        # Step 1: Fill interior holes
        filled = fill_holes(raw_mask)

    # Step 2: Morphological Opening
    opened = morphological_opening(filled, kernel_size=kernel_size)

    # Step 3: Morphological Closing
    closed = morphological_closing(opened, kernel_size=kernel_size)

    # Step 4: Small object noise removal
    cleaned = remove_small_objects(closed, min_area=min_area)

    # Step 5: Optional border object removal
    if remove_border:
        cleaned = remove_border_objects(cleaned)

    return cleaned
