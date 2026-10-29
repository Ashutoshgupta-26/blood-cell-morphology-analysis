"""
Visualization Utility Functions for Blood Smear Image, Preprocessing, and Segmentation Analysis.
Provides functions for side-by-side annotations, preprocessing comparison grids, color space plots,
6-panel segmentation analysis figures, GT box overlay validation plots, and method comparison grids.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to load config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    VISUALIZATIONS_DIR,
    PREPROCESSING_VIS_DIR,
    SEGMENTATION_VIS_DIR,
    ensure_directories,
    CLASS_COLORS_RGB,
    CLASS_COLORS_BGR,
    CLASS_ID_TO_NAME,
)
from .annotation_utils import parse_yolo_annotation, draw_ground_truth_boxes
from .preprocessing import (
    load_image,
    convert_color_spaces,
    grayscale_conversion,
    gaussian_blur,
    median_blur,
    bilateral_filter,
    histogram_equalization,
    clahe_enhancement,
    correct_illumination,
)
from .segmentation import (
    segment_cells,
    analyze_connected_components,
    validate_object_level_coverage,
    otsu_segmentation,
    adaptive_segmentation,
    color_based_segmentation,
    kmeans_segmentation,
)


def save_annotated_image(annotated_bgr, filename, output_dir=None):
    """
    Save an annotated OpenCV BGR image to the visualizations directory.
    """
    if output_dir is None:
        ensure_directories()
        output_dir = VISUALIZATIONS_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / filename
    cv2.imwrite(str(target_path), annotated_bgr)
    return target_path


def create_side_by_side_comparison(image_path, label_path, save_filename=None, output_dir=None):
    """
    Load an image and its YOLO annotations, then generate and save a side-by-side comparison figure.
    Displays Original Smear on the left and Ground-Truth Annotated Smear on the right.
    """
    image_path = Path(image_path)
    label_path = Path(label_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image file: {image_path}")

    img_h, img_w = img_bgr.shape[:2]

    annotations = []
    if label_path.exists():
        annotations = parse_yolo_annotation(label_path, img_w, img_h)

    annotated_bgr = draw_ground_truth_boxes(img_bgr, annotations)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Blood Smear ({img_w}x{img_h})\n{image_path.name}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(annotated_rgb)
    class_counts = {}
    for ann in annotations:
        cname = ann["class_name"]
        class_counts[cname] = class_counts.get(cname, 0) + 1
    
    counts_str = ", ".join([f"{name}: {cnt}" for name, cnt in sorted(class_counts.items())])
    axes[1].set_title(f"Ground-Truth Annotations ({len(annotations)} cells)\n{counts_str}", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()

    saved_path = None
    if save_filename:
        if output_dir is None:
            ensure_directories()
            output_dir = VISUALIZATIONS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        saved_path = output_dir / save_filename
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return annotated_bgr, annotations, saved_path


def create_preprocessing_comparison_grid(image_path, save_filename=None, output_dir=None):
    """
    Generate a 3x3 grid figure comparing 9 preprocessing methods/steps on a single image.
    """
    image_path = Path(image_path)
    img_bgr = load_image(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray_std = grayscale_conversion(img_bgr, method="standard")
    green_channel = grayscale_conversion(img_bgr, method="green")

    gauss = gaussian_blur(green_channel)
    median = median_blur(green_channel)
    bilateral = bilateral_filter(green_channel)

    hist_eq = histogram_equalization(green_channel)
    clahe_img = clahe_enhancement(green_channel)

    illum_corrected = correct_illumination(green_channel)
    clahe_illum = clahe_enhancement(illum_corrected)

    methods = [
        ("Original RGB", img_rgb, "color"),
        ("Standard Grayscale", gray_std, "gray"),
        ("Green Channel (High Contrast)", green_channel, "gray"),
        ("Gaussian Filter (5x5)", gauss, "gray"),
        ("Median Filter (5x5)", median, "gray"),
        ("Bilateral Filter (Edge-Preserving)", bilateral, "gray"),
        ("Global Hist Equalization", hist_eq, "gray"),
        ("CLAHE (Local Adaptive)", clahe_img, "gray"),
        ("Illum. Corrected + CLAHE", clahe_illum, "gray"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Preprocessing Method Comparison Grid\n{image_path.name}", fontsize=16, fontweight="bold")

    for idx, (title, img_arr, cmap_type) in enumerate(methods):
        ax = axes[idx // 3, idx % 3]
        if cmap_type == "color":
            ax.imshow(img_arr)
        else:
            ax.imshow(img_arr, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    saved_path = None
    if save_filename:
        if output_dir is None:
            ensure_directories()
            output_dir = PREPROCESSING_VIS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = output_dir / save_filename
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return saved_path


def create_color_space_comparison_figure(image_path, save_filename=None, output_dir=None):
    """
    Generate a 3x4 subplot figure decomposing a blood smear image into RGB, HSV, and LAB channels.
    """
    image_path = Path(image_path)
    img_bgr = load_image(image_path)
    cs = convert_color_spaces(img_bgr)

    grid = [
        [("Original RGB", cs["RGB"], "color"), ("RGB - Red Channel", cs["R_channel"], "Reds"),
         ("RGB - Green Channel", cs["G_channel"], "Greens"), ("RGB - Blue Channel", cs["B_channel"], "Blues")],
        [("Original RGB", cs["RGB"], "color"), ("HSV - Hue (H)", cs["H_channel"], "hsv"),
         ("HSV - Saturation (S)", cs["S_channel"], "gray"), ("HSV - Value (V)", cs["V_channel"], "gray")],
        [("Original RGB", cs["RGB"], "color"), ("LAB - Lightness (L)", cs["L_channel"], "gray"),
         ("LAB - 'a' (Green-Red)", cs["A_channel"], "PRGn"), ("LAB - 'b' (Blue-Yellow)", cs["B_lab_channel"], "YlGnBu")],
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(f"Color Space & Channel Decomposition Analysis\n{image_path.name}", fontsize=16, fontweight="bold")

    for r_idx, row in enumerate(grid):
        for c_idx, (title, img_arr, cmap_name) in enumerate(row):
            ax = axes[r_idx, c_idx]
            if cmap_name == "color":
                ax.imshow(img_arr)
            else:
                ax.imshow(img_arr, cmap=cmap_name)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    saved_path = None
    if save_filename:
        if output_dir is None:
            ensure_directories()
            output_dir = PREPROCESSING_VIS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = output_dir / save_filename
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return saved_path


def create_segmentation_comparison_figure(image_path, label_path=None, save_filename=None, output_dir=None):
    """
    Generate a 6-panel grid subplot figure displaying the complete Stage 3 segmentation pipeline:
    
    1. Original RGB Image
    2. Preprocessed Image (Green Channel + Bilateral Filter)
    3. Raw Binary Mask (Post-Otsu Thresholding)
    4. Morphologically Cleaned Mask (Post-Hole Filling & Small Object Removal)
    5. Connected Components Visualization (Color-Mapped Labels)
    6. Ground-Truth Bounding Box Overlay with Segmentation Contours
    
    Args:
        image_path (str or Path): Path to blood smear image file.
        label_path (str or Path, optional): Path to YOLO annotation text file.
        save_filename (str, optional): Filename for saved plot figure.
        output_dir (Path, optional): Directory to save figure. Defaults to SEGMENTATION_VIS_DIR.
        
    Returns:
        Path: Path to saved figure.
    """
    image_path = Path(image_path)
    img_bgr = load_image(image_path)
    h, w = img_bgr.shape[:2]

    # Run Stage 3 segmentation pipeline
    seg_res = segment_cells(img_bgr, method="otsu", channel="green")
    preprocessed_img = seg_res["preprocessed_img"]
    raw_mask = seg_res["raw_mask"]
    cleaned_mask = seg_res["cleaned_mask"]
    labeled_mask = seg_res["labeled_mask"]
    num_components = seg_res["num_components"]
    threshold_val = seg_res["otsu_threshold"]

    # Parse GT annotations if label_path exists
    annotations = []
    if label_path and Path(label_path).exists():
        annotations = parse_yolo_annotation(label_path, w, h)

    # Validation coverage
    val_res = validate_object_level_coverage(annotations, cleaned_mask)

    # 1. RGB Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Color-mapped Connected Components Visualization
    # Assign distinct random colors to each component label
    label_colors = np.random.randint(50, 255, size=(num_components + 1, 3), dtype=np.uint8)
    label_colors[0] = [0, 0, 0]  # Background is black
    cc_vis_rgb = label_colors[labeled_mask]

    # 3. Overlay Ground-Truth Boxes & Segmentation Contours
    overlay_bgr = draw_ground_truth_boxes(img_bgr, annotations)
    # Extract contours of cleaned binary mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw cyan contours around detected cell regions
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 0), 1)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        f"Stage 3 Cell Segmentation & GT Overlay Analysis\n{image_path.name} ({w}x{h} px)",
        fontsize=15,
        fontweight="bold",
    )

    # Subplot 1: Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Original RGB Smear", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Subplot 2: Preprocessed
    axes[0, 1].imshow(preprocessed_img, cmap="gray")
    axes[0, 1].set_title("2. Preprocessed (Green Channel + Bilateral)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Subplot 3: Raw Binary Mask
    axes[0, 2].imshow(raw_mask, cmap="gray")
    thresh_str = f" (T={threshold_val:.1f})" if threshold_val else ""
    axes[0, 2].set_title(f"3. Raw Otsu Mask{thresh_str}", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # Subplot 4: Cleaned Mask
    axes[1, 0].imshow(cleaned_mask, cmap="gray")
    axes[1, 0].set_title("4. Morphologically Cleaned Mask", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Subplot 5: Connected Components
    axes[1, 1].imshow(cc_vis_rgb)
    axes[1, 1].set_title(f"5. Connected Components ({num_components} regions)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Subplot 6: Ground-Truth Overlay
    axes[1, 2].imshow(overlay_rgb)
    cov_pct = val_res["coverage_rate_pct"]
    num_gt = val_res["total_gt_boxes"]
    axes[1, 2].set_title(f"6. GT Boxes Overlay ({cov_pct}% Box Coverage)\nGT Boxes: {num_gt} | Cyan: Detected Mask", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)

    saved_path = None
    if save_filename:
        if output_dir is None:
            ensure_directories()
            output_dir = SEGMENTATION_VIS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = output_dir / save_filename
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return saved_path


def create_segmentation_method_comparison_figure(image_path, save_filename=None, output_dir=None):
    """
    Generate a 2x2 subplot figure comparing 4 segmentation algorithms on the same image:
    1. Global Otsu Thresholding
    2. Local Adaptive Gaussian Thresholding
    3. Color-Based HSV Saturation Thresholding
    4. Classical K-Means Clustering (k=3)
    
    Args:
        image_path (str or Path): Path to blood smear image file.
        save_filename (str, optional): Filename for saved plot figure.
        output_dir (Path, optional): Directory to save figure. Defaults to SEGMENTATION_VIS_DIR.
        
    Returns:
        Path: Path to saved figure.
    """
    image_path = Path(image_path)
    img_bgr = load_image(image_path)

    # 1. Otsu
    res_otsu = segment_cells(img_bgr, method="otsu", channel="green")
    # 2. Adaptive
    res_adap = segment_cells(img_bgr, method="adaptive", channel="green")
    # 3. Color Range
    res_color = segment_cells(img_bgr, method="color_range", channel="hsv_s")
    # 4. K-Means
    res_kmeans = segment_cells(img_bgr, method="kmeans", channel="green")

    grid = [
        ("Otsu Thresholding", res_otsu["cleaned_mask"], res_otsu["num_components"]),
        ("Adaptive Gaussian Thresholding", res_adap["cleaned_mask"], res_adap["num_components"]),
        ("Color-Based HSV Thresholding", res_color["cleaned_mask"], res_color["num_components"]),
        ("K-Means Clustering (k=3)", res_kmeans["cleaned_mask"], res_kmeans["num_components"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Segmentation Algorithm Comparison\n{image_path.name}", fontsize=15, fontweight="bold")

    for idx, (title, mask, num_comp) in enumerate(grid):
        ax = axes[idx // 2, idx % 2]
        ax.imshow(mask, cmap="gray")
        ax.set_title(f"{title}\n({num_comp} components detected)", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)

    saved_path = None
    if save_filename:
        if output_dir is None:
            ensure_directories()
            output_dir = SEGMENTATION_VIS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = output_dir / save_filename
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return saved_path


def create_watershed_visualization(image_path, seg_res, watershed_res, save_filename=None, output_dir=None):
    """
    Generate a 2x3 subplot figure displaying the Stage 4 Watershed results:
    1. Original Image (RGB)
    2. Cleaned binary mask (Stage 3)
    3. Connected components BEFORE watershed
    4. Distance Transform and Markers
    5. Watershed separated instances
    6. Final contours and instance IDs overlaid on original image
    """
    import matplotlib.colors as mcolors
    from config import WATERSHED_VIS_DIR
    
    image_path = Path(image_path)
    img_bgr = load_image(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cleaned_mask = seg_res["cleaned_mask"]
    cc_before = seg_res["labeled_mask"]
    num_before = seg_res["num_components"]

    dist_transform = watershed_res["dist_transform"]
    markers = watershed_res["markers"]
    labels = watershed_res["labels"]
    num_cells = labels.max()

    # Stable colors make before/after figures easier to compare between runs.
    rng = np.random.default_rng(0)
    if num_cells > 0:
        colors = rng.random((num_cells, 3))
        colors = np.vstack([[0, 0, 0], colors]) # Black for background
        cmap_after = mcolors.ListedColormap(colors)
    else:
        cmap_after = 'gray'
        
    if num_before > 0:
        colors_b = rng.random((num_before, 3))
        colors_b = np.vstack([[0, 0, 0], colors_b])
        cmap_before = mcolors.ListedColormap(colors_b)
    else:
        cmap_before = 'gray'

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Stage 4: Watershed Segmentation Analysis\n{image_path.name}", fontsize=15, fontweight="bold")

    # 1. Original Image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Original RGB Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Stage 3 Cleaned Mask
    axes[0, 1].imshow(cleaned_mask, cmap="gray")
    axes[0, 1].set_title("2. Cleaned Mask (Stage 3)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Connected Components Before
    axes[0, 2].imshow(cc_before, cmap=cmap_before, interpolation="nearest")
    axes[0, 2].set_title(f"3. CC Before Watershed ({num_before} objects)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # 4. Distance Transform + Markers
    # Overlay markers on distance transform
    axes[1, 0].imshow(dist_transform, cmap="viridis")
    axes[1, 0].imshow(np.ma.masked_where(markers == 0, markers), cmap='autumn', alpha=0.5)
    axes[1, 0].set_title("4. Distance Transform + Markers (Red)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # 5. Watershed Instances
    axes[1, 1].imshow(labels, cmap=cmap_after, interpolation="nearest")
    axes[1, 1].set_title(f"5. Instances After Watershed ({num_cells} distinct cells)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # 6. Final Contours over Original Image
    contour_img = img_bgr.copy()
    from .watershed import extract_instance_contours
    contours_dict = extract_instance_contours(labels)
    for l_id, contour in contours_dict.items():
        # Draw contour
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 255), 1)
        # Put ID text
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(contour_img, str(l_id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
    axes[1, 2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Final Contours & Instance IDs", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    saved_path = None
    if save_filename:
        if output_dir is None:
            ensure_directories()
            output_dir = WATERSHED_VIS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = output_dir / save_filename
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return saved_path
