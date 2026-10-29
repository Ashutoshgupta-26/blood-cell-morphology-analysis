"""
Annotation Utility Functions for YOLO Annotation Parsing and Bounding Box Drawing.
Handles conversion of normalized coordinates (cx, cy, w, h) to pixel bounding boxes (x1, y1, x2, y2),
validates malformed label lines, and draws colored ground-truth annotations onto images.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path to load config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_ID_TO_NAME, CLASS_COLORS_BGR, load_dataset_classes


def yolo_to_pixel_coords(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    """
    Convert normalized YOLO coordinates (0.0 to 1.0) into pixel coordinates.
    
    Args:
        x_center_norm (float): Bounding box center X (normalized 0..1).
        y_center_norm (float): Bounding box center Y (normalized 0..1).
        width_norm (float): Bounding box width (normalized 0..1).
        height_norm (float): Bounding box height (normalized 0..1).
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.
        
    Returns:
        tuple: (x1, y1, x2, y2) in pixel integer coordinates, clipped to image boundary.
    """
    # Calculate box width and height in pixels
    w_px = width_norm * img_width
    h_px = height_norm * img_height

    # Calculate top-left (x1, y1) and bottom-right (x2, y2)
    x1 = (x_center_norm - width_norm / 2.0) * img_width
    y1 = (y_center_norm - height_norm / 2.0) * img_height
    x2 = x1 + w_px
    y2 = y1 + h_px

    # Clip to image boundary
    x1_clipped = max(0, min(int(round(x1)), img_width - 1))
    y1_clipped = max(0, min(int(round(y1)), img_height - 1))
    x2_clipped = max(0, min(int(round(x2)), img_width - 1))
    y2_clipped = max(0, min(int(round(y2)), img_height - 1))

    return x1_clipped, y1_clipped, x2_clipped, y2_clipped


def parse_yolo_annotation(label_path, img_width, img_height, class_map=None):
    """
    Parse a YOLO .txt annotation file and convert to pixel coordinates.
    
    Args:
        label_path (str or Path): Path to YOLO .txt file.
        img_width (int): Width of corresponding image in pixels.
        img_height (int): Height of corresponding image in pixels.
        class_map (dict, optional): Dictionary mapping class ID -> class name.
        
    Returns:
        list of dict: List of annotation objects with normalized & pixel coordinates and class names.
    """
    if class_map is None:
        class_map = CLASS_ID_TO_NAME

    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {label_path}")

    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Malformed annotation in {label_path.name} line {line_num}: expected 5 elements, got {len(parts)}. Line skipped.")
                continue

            try:
                class_id = int(parts[0])
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Validate normalized values are within reasonable bounds [0.0, 1.0]
                if not (0.0 <= cx_norm <= 1.0 and 0.0 <= cy_norm <= 1.0 and 0.0 <= w_norm <= 1.0 and 0.0 <= h_norm <= 1.0):
                    print(f"Warning: Normalized coordinates out of [0, 1] range in {label_path.name} line {line_num}. Clipping.")

                x1, y1, x2, y2 = yolo_to_pixel_coords(
                    cx_norm, cy_norm, w_norm, h_norm, img_width, img_height
                )

                class_name = class_map.get(class_id, f"Class_{class_id}")

                annotations.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "cx_norm": cx_norm,
                    "cy_norm": cy_norm,
                    "w_norm": w_norm,
                    "h_norm": h_norm,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width_px": x2 - x1,
                    "height_px": y2 - y1,
                })

            except ValueError as ve:
                print(f"Warning: Invalid numerical data in {label_path.name} line {line_num}: {ve}. Line skipped.")
                continue

    return annotations


def draw_ground_truth_boxes(image_bgr, annotations, class_colors=None, thickness=2, font_scale=0.5):
    """
    Draw ground-truth bounding boxes and class labels onto an OpenCV BGR image array.
    
    Args:
        image_bgr (np.ndarray): Image array in BGR format.
        annotations (list of dict): Annotation structures returned by parse_yolo_annotation.
        class_colors (dict, optional): Color map for class IDs in BGR format.
        thickness (int): Bounding box border thickness.
        font_scale (float): Text label font scale.
        
    Returns:
        np.ndarray: Annotated copy of the input BGR image.
    """
    if class_colors is None:
        class_colors = CLASS_COLORS_BGR

    annotated = image_bgr.copy()

    for ann in annotations:
        cid = ann["class_id"]
        cname = ann["class_name"]
        x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]

        color = class_colors.get(cid, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label text
        label_str = f"{cname}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Ensure label background rectangle stays inside image boundaries
        text_bg_y1 = max(0, y1 - text_height - 6)
        text_bg_y2 = max(text_height + 6, y1)

        # Draw solid label background rectangle for high legibility
        cv2.rectangle(
            annotated,
            (x1, text_bg_y1),
            (x1 + text_width + 4, text_bg_y2),
            color,
            -1  # Filled
        )

        # Draw white or black text depending on background brightness
        text_color = (255, 255, 255)
        cv2.putText(
            annotated,
            label_str,
            (x1 + 2, text_bg_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA
        )

    return annotated
