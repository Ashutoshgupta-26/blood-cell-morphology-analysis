"""Distance-transform watershed helpers for splitting touching cell regions."""

import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from config import WATERSHED_CONFIG


def compute_distance_transform(binary_mask):
    """Return the Euclidean distance from each foreground pixel to background."""
    foreground = np.asarray(binary_mask) > 0
    return cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)


def _find_peak_coordinates(distance_map, mask, min_distance, threshold_ratio):
    """Find stable cell-center seeds independently inside each connected region."""
    component_count, component_labels = cv2.connectedComponents(
        np.asarray(mask > 0, dtype=np.uint8), connectivity=8
    )
    coordinates = []
    for component_id in range(1, component_count):
        component = component_labels == component_id
        values = distance_map[component]
        if values.size == 0:
            continue
        peak_height = float(values.max())
        if peak_height <= 0:
            continue

        threshold_abs = peak_height * float(threshold_ratio)
        found = peak_local_max(
            distance_map,
            min_distance=max(1, int(min_distance)),
            threshold_abs=threshold_abs,
            labels=component.astype(np.uint8),
            exclude_border=False,
        )
        if not len(found):
            # A fallback seed prevents a foreground region from disappearing.
            local = np.where(component, distance_map, -1)
            found = np.asarray([np.unravel_index(int(np.argmax(local)), local.shape)])
        coordinates.extend((int(y), int(x)) for y, x in found)
    return coordinates


def generate_foreground_markers(dist_transform, binary_mask, min_distance=15, threshold_ratio=None):
    """Generate one marker at each accepted local distance-transform maximum."""
    if threshold_ratio is None:
        threshold_ratio = WATERSHED_CONFIG.get("distance_threshold_ratio", 0.25)
    distance_map = cv2.GaussianBlur(np.asarray(dist_transform, dtype=np.float32), (5, 5), 0)
    coordinates = _find_peak_coordinates(
        distance_map, binary_mask, min_distance, threshold_ratio
    )
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    sure_foreground = np.zeros(distance_map.shape, dtype=np.uint8)
    for marker_id, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = marker_id
        cv2.circle(sure_foreground, (x, y), 2, 255, -1)
    return sure_foreground, markers, len(coordinates) + 1


def generate_background_markers(binary_mask, kernel_size=(3, 3), iterations=3):
    """Return a dilated foreground envelope for visualization/debugging."""
    foreground = np.asarray(binary_mask > 0, dtype=np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.dilate(foreground, kernel, iterations=max(1, int(iterations)))


def separate_touching_cells(binary_mask, image_bgr=None, threshold_ratio=None):
    """Split touching binary foreground regions with distance-marker watershed.

    The input mask defines the foreground domain. Local maxima in the distance
    transform seed a watershed over that domain, so the algorithm cannot grow
    labels into plasma background. ``threshold_ratio`` controls the minimum
    peak height relative to the tallest peak in each connected foreground
    region; it does not remove foreground pixels based on a global threshold.
    """
    foreground = np.asarray(binary_mask) > 0
    if foreground.ndim != 2:
        raise ValueError("binary_mask must be a 2D single-channel image")
    h, w = foreground.shape
    if image_bgr is None:
        image_bgr = cv2.cvtColor(foreground.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[:2] != foreground.shape:
        raise ValueError("image_bgr and binary_mask must have matching image dimensions")

    ratio = threshold_ratio
    if ratio is None:
        ratio = WATERSHED_CONFIG.get("distance_threshold_ratio", 0.25)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    reference_distance = WATERSHED_CONFIG.get("min_peak_distance", 40)
    reference_short_edge = WATERSHED_CONFIG.get("reference_short_edge", 480)
    image_scale = min(h, w) / float(reference_short_edge)
    min_distance = max(6, int(round(reference_distance * image_scale)))
    min_area = max(1, int(WATERSHED_CONFIG.get("min_cell_area", 20)))

    # Treat pixels beyond the image frame as background. Without this border,
    # distance peaks for edge-clipped RBCs are biased and can merge with a
    # neighboring cell or be split at an arbitrary point.
    padded_foreground = np.pad(foreground, 1, mode="constant", constant_values=False)
    padded_distance = compute_distance_transform(padded_foreground)
    sure_foreground_padded, seed_markers_padded, marker_count = generate_foreground_markers(
        padded_distance, padded_foreground, min_distance=min_distance, threshold_ratio=ratio
    )
    if marker_count <= 1 or not foreground.any():
        labels = np.zeros((h, w), dtype=np.int32)
        distance_map = padded_distance[1:-1, 1:-1]
        sure_foreground = sure_foreground_padded[1:-1, 1:-1]
        seed_markers = seed_markers_padded[1:-1, 1:-1]
    else:
        padded_labels = watershed(
            -padded_distance,
            seed_markers_padded,
            mask=padded_foreground,
            watershed_line=True,
        )
        labels = padded_labels[1:-1, 1:-1]
        distance_map = padded_distance[1:-1, 1:-1]
        sure_foreground = sure_foreground_padded[1:-1, 1:-1]
        seed_markers = seed_markers_padded[1:-1, 1:-1]
        # Reindex after removing tiny watershed fragments, keeping IDs contiguous.
        filtered = np.zeros_like(labels, dtype=np.int32)
        next_id = 1
        for old_id in np.unique(labels):
            if old_id <= 0:
                continue
            region = labels == old_id
            if int(region.sum()) >= min_area:
                filtered[region] = next_id
                next_id += 1
        labels = filtered

    sure_background = generate_background_markers(foreground)
    debug_overlay = image_bgr.copy()
    boundary = cv2.morphologyEx(
        (labels > 0).astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    debug_overlay[boundary] = (0, 0, 255)
    debug_overlay[sure_foreground > 0] = (0, 255, 0)

    return {
        "labels": labels,
        "num_instances": int(labels.max()),
        "debug_overlay": debug_overlay,
        "distance_transform": distance_map,
        "dist_transform": distance_map,
        "markers": seed_markers,
        "sure_fg": sure_foreground,
        "sure_bg": sure_background,
        "raw_watershed_labels": labels,
        "num_markers": marker_count - 1,
        "threshold_ratio": ratio,
    }


def extract_instance_contours(labels):
    """Return the largest external contour for each nonempty instance label."""
    contours_by_label = {}
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        mask = (labels == instance_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours_by_label[int(instance_id)] = max(contours, key=cv2.contourArea)
    return contours_by_label


def compute_instance_statistics(labels, image_name):
    """Compute area, bounding box, centroid, and perimeter for each instance."""
    rows = []
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        mask = (labels == instance_id).astype(np.uint8) * 255
        area = int(cv2.countNonZero(mask))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x, center_y = x + width / 2, y + height / 2
        rows.append({
            "image_name": image_name,
            "instance_id": int(instance_id),
            "area": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": width,
            "bbox_height": height,
            "centroid_x": center_x,
            "centroid_y": center_y,
            "perimeter": cv2.arcLength(contour, True),
        })
    return pd.DataFrame(rows, columns=[
        "image_name", "instance_id", "area", "bbox_x", "bbox_y",
        "bbox_width", "bbox_height", "centroid_x", "centroid_y", "perimeter",
    ])


def analyze_gt_overlap(labels, annotations, image_name):
    """Summarize instance presence inside GT rectangles; this is not mask IoU."""
    rows = []
    height, width = labels.shape
    for ann in annotations:
        x1 = int(np.clip(ann["x1"], 0, width))
        y1 = int(np.clip(ann["y1"], 0, height))
        x2 = int(np.clip(ann["x2"], x1, width))
        y2 = int(np.clip(ann["y2"], y1, height))
        crop = labels[y1:y2, x1:x2]
        ids, counts = np.unique(crop[crop > 0], return_counts=True)
        box_area = max(1, crop.size)
        rows.append({
            "image_name": image_name,
            "gt_class": ann["class_name"],
            "gt_x": x1,
            "gt_y": y1,
            "gt_width": x2 - x1,
            "gt_height": y2 - y1,
            "num_watershed_instances_overlapping": int(len(ids)),
            "box_foreground_fraction": round(float(np.sum(counts) / box_area), 4),
            "largest_instance_fraction_in_box": round(float(counts.max() / box_area), 4) if len(counts) else 0.0,
        })
    return pd.DataFrame(rows, columns=[
        "image_name", "gt_class", "gt_x", "gt_y", "gt_width", "gt_height",
        "num_watershed_instances_overlapping", "box_foreground_fraction",
        "largest_instance_fraction_in_box",
    ])
