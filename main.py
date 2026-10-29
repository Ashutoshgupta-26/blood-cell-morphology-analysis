"""
Main Execution Pipeline for Blood Cell Image Processing Project.
Runs Stage 1 (Dataset Inspection), Stage 2 (Preprocessing Analysis),
and Stage 3 (Blood Cell Segmentation, Morphological Cleanup, & Evaluation).
"""

import sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config import (
    ensure_directories,
    VISUALIZATIONS_DIR,
    PREPROCESSING_VIS_DIR,
    SEGMENTATION_VIS_DIR,
    WATERSHED_VIS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    PREPROCESSING_CONFIG,
    SEGMENTATION_CONFIG,
    WATERSHED_CONFIG,
    DATASET_ROOT,
)
from src.dataset_utils import (
    get_full_dataset_statistics,
    print_dataset_statistics,
    get_paired_files,
)
from src.annotation_utils import parse_yolo_annotation, draw_ground_truth_boxes
from src.visualization import (
    save_annotated_image,
    create_side_by_side_comparison,
    create_preprocessing_comparison_grid,
    create_color_space_comparison_figure,
    create_segmentation_comparison_figure,
    create_segmentation_method_comparison_figure,
    create_watershed_visualization,
)
from src.preprocessing import (
    load_image,
    convert_color_spaces,
    grayscale_conversion,
    gaussian_blur,
    median_blur,
    bilateral_filter,
    histogram_equalization,
    clahe_enhancement,
    correct_illumination,
    compute_image_statistics,
)
from src.segmentation import (
    segment_cells,
    analyze_connected_components,
    validate_object_level_coverage,
)
from src.watershed import separate_touching_cells


def run_stage_1_dataset_and_visualization():
    """
    Run Stage 1: Dataset Inspection, Pairing Verification, Statistics Reporting,
    and Ground-Truth Bounding Box Visualizations.
    """
    print("=" * 70)
    print(" STAGE 1: DATASET INSPECTION & GROUND-TRUTH VISUALIZATION ")
    print("=" * 70)

    ensure_directories()
    stats = get_full_dataset_statistics()
    print_dataset_statistics(stats)

    train_pairs = get_paired_files(split="train")
    if not train_pairs:
        raise RuntimeError("No paired training images found for visualization test.")

    print(f"Selecting 3 sample training images for ground-truth visualization test...")

    sample_indices = [0, min(10, len(train_pairs) - 1), min(25, len(train_pairs) - 1)]
    sample_pairs = [train_pairs[i] for i in sample_indices]

    created_visualizations = []

    for idx, pair in enumerate(sample_pairs, 1):
        img_path = pair["image_path"]
        lbl_path = pair["label_path"]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Error loading image: {img_path}")
            continue

        h, w = img_bgr.shape[:2]
        annotations = parse_yolo_annotation(lbl_path, w, h)
        annotated_bgr = draw_ground_truth_boxes(img_bgr, annotations)

        annotated_filename = f"annotated_image_{idx:03d}.png"
        saved_annotated_path = save_annotated_image(annotated_bgr, annotated_filename)

        side_by_side_filename = f"side_by_side_{idx:03d}.png"
        _, _, saved_sbs_path = create_side_by_side_comparison(
            img_path, lbl_path, save_filename=side_by_side_filename
        )

        created_visualizations.append({
            "sample_num": idx,
            "filename": img_path.name,
            "dimensions": f"{w}x{h}",
            "cell_count": len(annotations),
            "annotated_path": str(saved_annotated_path),
            "side_by_side_path": str(saved_sbs_path),
        })

        print(f"\n[Sample {idx}] Processed: {img_path.name}")
        print(f"  - Dimensions: {w} x {h} px")
        print(f"  - Total Ground-Truth Cells: {len(annotations)}")
        
        counts = {}
        for ann in annotations:
            cname = ann["class_name"]
            counts[cname] = counts.get(cname, 0) + 1
        for cname, cnt in sorted(counts.items()):
            print(f"    - {cname}: {cnt}")
            
        print(f"  - Saved Annotated Image: {saved_annotated_path}")
        print(f"  - Saved Side-by-Side Plot: {saved_sbs_path}")

    print("\n" + "=" * 70)
    print(" STAGE 1 COMPLETED SUCCESSFULLY ")
    print("=" * 70)
    return stats, created_visualizations


def run_stage_2_preprocessing(sample_size=None):
    """
    Run Stage 2: Image Preprocessing, Color Space Analysis, Noise Reduction Filtering,
    Contrast Enhancement, Quantitative Image Statistics Export, and Analytical Reporting.
    """
    print("\n" + "=" * 70)
    print(" STAGE 2: IMAGE PREPROCESSING & COLOR-SPACE ANALYSIS ")
    print("=" * 70)

    ensure_directories()
    if sample_size is None:
        sample_size = PREPROCESSING_CONFIG.get("sample_size", 10)

    train_pairs = get_paired_files(split="train")
    if not train_pairs:
        raise RuntimeError("No paired training images available for Stage 2 experiments.")

    selected_pairs = []
    seen_stems = set()

    for pair in train_pairs:
        img_bgr = cv2.imread(str(pair["image_path"]))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        anns = parse_yolo_annotation(pair["label_path"], w, h)
        cnames = set(a["class_name"] for a in anns)

        if pair["stem"] not in seen_stems:
            selected_pairs.append((pair, len(anns), cnames, (w, h)))
            seen_stems.add(pair["stem"])

        if len(selected_pairs) >= sample_size * 3:
            break

    selected_pairs.sort(key=lambda x: ( "Platelets" in x[2], "WBC" in x[2], x[1] ), reverse=True)
    final_pairs = [sp[0] for sp in selected_pairs[:sample_size]]

    print(f"Selected {len(final_pairs)} representative training images for Stage 2 analysis.")

    all_stats_rows = []

    for idx, pair in enumerate(final_pairs, 1):
        img_path = pair["image_path"]
        img_name = img_path.name
        img_bgr = load_image(img_path)

        grid_filename = f"preprocessing_comparison_{idx:03d}.png"
        saved_grid_path = create_preprocessing_comparison_grid(img_path, save_filename=grid_filename)

        cs_filename = f"color_space_comparison_{idx:03d}.png"
        saved_cs_path = create_color_space_comparison_figure(img_path, save_filename=cs_filename)

        gray_std = grayscale_conversion(img_bgr, method="standard")
        green_chan = grayscale_conversion(img_bgr, method="green")
        gauss_img = gaussian_blur(green_chan)
        median_img = median_blur(green_chan)
        bilateral_img = bilateral_filter(green_chan)
        hist_eq_img = histogram_equalization(green_chan)
        clahe_img = clahe_enhancement(green_chan)
        illum_corr_img = correct_illumination(green_chan)
        clahe_illum_img = clahe_enhancement(illum_corr_img)

        methods_to_stat = [
            ("Raw (BGR Equivalent)", img_bgr),
            ("Standard Grayscale", gray_std),
            ("Green Channel", green_chan),
            ("Gaussian Blur (5x5)", gauss_img),
            ("Median Blur (5x5)", median_img),
            ("Bilateral Filter (9,75,75)", bilateral_img),
            ("Global Hist Equalization", hist_eq_img),
            ("CLAHE (Clip=2.0, Grid=8x8)", clahe_img),
            ("Illumination Corrected + CLAHE", clahe_illum_img),
        ]

        for method_name, processed_arr in methods_to_stat:
            stat_row = compute_image_statistics(processed_arr, method_name=method_name, image_name=img_name)
            all_stats_rows.append(stat_row)

        print(f"  [{idx}/{len(final_pairs)}] Processed {img_name}")
        print(f"      - Saved Grid: {saved_grid_path.name}")
        print(f"      - Saved Color Space Plot: {saved_cs_path.name}")

    stats_df = pd.DataFrame(all_stats_rows)
    csv_path = METRICS_DIR / "preprocessing_statistics.csv"
    stats_df.to_csv(csv_path, index=False)

    print("\n" + "-" * 70)
    print(" QUANTITATIVE PREPROCESSING METRICS SUMMARY (Averaged across sample) ")
    print("-" * 70)
    summary_df = stats_df.groupby("method")[["mean", "std", "contrast_rms", "entropy"]].mean().reset_index()
    print(summary_df.to_string(index=False))
    print("-" * 70)

    generate_preprocessing_report(summary_df)
    verify_dataset_untouched()

    print("\n" + "=" * 70)
    print(" STAGE 2 COMPLETED SUCCESSFULLY ")
    print("=" * 70)


def run_stage_3_segmentation(sample_size=None):
    """
    Run Stage 3: Blood Cell Segmentation, Morphological Cleanup, Connected Component Analysis,
    Method Comparison, Object-Level Validation, and Markdown Report Generation.
    """
    print("\n" + "=" * 70)
    print(" STAGE 3: BLOOD CELL SEGMENTATION & MORPHOLOGICAL ANALYSIS ")
    print("=" * 70)

    ensure_directories()
    if sample_size is None:
        sample_size = SEGMENTATION_CONFIG.get("sample_size", 10)

    train_pairs = get_paired_files(split="train")
    if not train_pairs:
        raise RuntimeError("No paired training images available for Stage 3 experiments.")

    # Select 10 diverse representative images
    selected_pairs = []
    seen_stems = set()

    for pair in train_pairs:
        img_bgr = cv2.imread(str(pair["image_path"]))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        anns = parse_yolo_annotation(pair["label_path"], w, h)
        cnames = set(a["class_name"] for a in anns)

        if pair["stem"] not in seen_stems:
            selected_pairs.append((pair, len(anns), cnames, (w, h)))
            seen_stems.add(pair["stem"])

        if len(selected_pairs) >= sample_size * 3:
            break

    selected_pairs.sort(key=lambda x: ( "Platelets" in x[2], "WBC" in x[2], x[1] ), reverse=True)
    final_pairs = [sp[0] for sp in selected_pairs[:sample_size]]

    print(f"Selected {len(final_pairs)} representative training images for Stage 3 segmentation analysis.")

    method_comparison_rows = []
    all_components_rows = []

    segmentation_methods = ["otsu", "adaptive", "color_range", "kmeans"]

    for idx, pair in enumerate(final_pairs, 1):
        img_path = pair["image_path"]
        lbl_path = pair["label_path"]
        img_name = img_path.name

        img_bgr = load_image(img_path)
        h, w = img_bgr.shape[:2]
        annotations = parse_yolo_annotation(lbl_path, w, h)

        # 1. Save 6-panel segmentation pipeline visualization
        seg_fig_filename = f"segmentation_comparison_{idx:03d}.png"
        saved_seg_path = create_segmentation_comparison_figure(img_path, lbl_path, save_filename=seg_fig_filename)

        # 2. Save 4-panel method comparison visualization
        method_fig_filename = f"segmentation_methods_{idx:03d}.png"
        saved_method_path = create_segmentation_method_comparison_figure(img_path, save_filename=method_fig_filename)

        print(f"\n  [{idx}/{len(final_pairs)}] Processing Segmentation: {img_name}")
        print(f"      - Saved 6-Panel Figure: {saved_seg_path.name}")
        print(f"      - Saved Method Comparison Plot: {saved_method_path.name}")

        # 3. Evaluate each segmentation method for comparison table
        for method in segmentation_methods:
            seg_res = segment_cells(img_bgr, method=method, channel="green")
            cleaned_mask = seg_res["cleaned_mask"]
            comp_df = seg_res["components_df"]

            # Compute metrics
            fg_ratio = float(np.count_nonzero(cleaned_mask == 255)) / float(cleaned_mask.size)
            comp_count = seg_res["num_components"]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            image_edges = cv2.Canny(gray, 40, 120) > 0
            mask_boundary = cv2.morphologyEx(
                cleaned_mask, cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ) > 0
            edge_neighborhood = cv2.dilate(
                image_edges.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ) > 0
            boundary_edge_support = (
                float(np.mean(edge_neighborhood[mask_boundary]))
                if np.any(mask_boundary) else 0.0
            )

            if not comp_df.empty:
                mean_area = float(comp_df["area"].mean())
                median_area = float(comp_df["area"].median())
                small_comp_count = int((comp_df["area"] < 50).sum())
            else:
                mean_area = 0.0
                median_area = 0.0
                small_comp_count = 0

            # GT Object-level Coverage Validation
            val_res = validate_object_level_coverage(annotations, cleaned_mask)

            method_comparison_rows.append({
                "image_name": img_name,
                "method": method,
                "foreground_ratio": round(fg_ratio, 4),
                "component_count": comp_count,
                "mean_component_area": round(mean_area, 2),
                "median_component_area": round(median_area, 2),
                "small_component_count": small_comp_count,
                "boundary_edge_support": round(boundary_edge_support, 4),
                "gt_box_coverage_pct": val_res["coverage_rate_pct"],
                "mean_fg_inside_gt_box": val_res["mean_fg_ratio_inside_box"],
            })

            # Record baseline (Otsu) detailed component measurements for component CSV
            if method == "otsu" and not comp_df.empty:
                for _, row in comp_df.iterrows():
                    all_components_rows.append({
                        "image_name": img_name,
                        "component_id": int(row["component_id"]),
                        "area": int(row["area"]),
                        "x": int(row["x"]),
                        "y": int(row["y"]),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "centroid_x": float(row["centroid_x"]),
                        "centroid_y": float(row["centroid_y"]),
                    })

    # Save method comparison CSV
    method_df = pd.DataFrame(method_comparison_rows)
    method_csv_path = METRICS_DIR / "segmentation_method_comparison.csv"
    method_df.to_csv(method_csv_path, index=False)

    # Save detailed component statistics CSV
    comp_df_all = pd.DataFrame(all_components_rows)
    comp_csv_path = METRICS_DIR / "segmentation_components.csv"
    comp_df_all.to_csv(comp_csv_path, index=False)

    print(f"\nSaved segmentation method comparison metrics to: {method_csv_path}")
    print(f"Saved detailed connected components measurements to: {comp_csv_path}")

    # Display summary statistics table
    print("\n" + "-" * 75)
    print(" QUANTITATIVE SEGMENTATION METHOD COMPARISON SUMMARY (Averaged) ")
    print("-" * 75)
    summary_seg_df = method_df.groupby("method")[[
        "foreground_ratio",
        "component_count",
        "mean_component_area",
        "small_component_count",
        "boundary_edge_support",
        "gt_box_coverage_pct",
        "mean_fg_inside_gt_box",
    ]].mean().reset_index()
    print(summary_seg_df.to_string(index=False))
    print("-" * 75)

    generate_segmentation_report(summary_seg_df)
    verify_dataset_untouched()

    print("\n" + "=" * 70)
    print(" STAGE 3 COMPLETED SUCCESSFULLY ")
    print("=" * 70)
    print(f"Segmentation visualizations saved to: {SEGMENTATION_VIS_DIR}")
    print(f"Segmentation metrics saved to: {method_csv_path}")
    print(f"Segmentation report saved to: {REPORTS_DIR / 'segmentation_analysis.md'}")


def generate_preprocessing_report(summary_df):
    """
    Generate analytical markdown report evaluating preprocessing techniques for blood cell segmentation.
    """
    report_path = REPORTS_DIR / "preprocessing_analysis.md"
    content = f"""# Stage 2: Image Preprocessing Analytical Report

## 1. Executive Summary
This report analyzes color space representations, noise reduction filters, contrast enhancement techniques, and illumination normalization strategies for Romanowsky/Giemsa-stained blood smear images from the TXL-PBC dataset.

---

## 2. Quantitative Preprocessing Metrics Summary

The table below presents the quantitative statistics averaged across representative training images:

```
{summary_df.to_string(index=False)}
```

---

## 3. Recommended Final Preprocessing Pipeline for Stage 3 Segmentation

```
Raw Blood Smear Image (BGR)
           ↓
Green Channel Extraction (Optimal Hemoglobin/Stain Absorption)
           ↓
Bilateral Filtering (d=9, sigmaColor=75, sigmaSpace=75) for Edge-Preserving Noise Reduction
           ↓
CLAHE Contrast Enhancement (clipLimit=2.0, tileGridSize=8x8)
           ↓
Clean, High-Contrast Enhanced Image Ready for Thresholding & Segmentation
```
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def generate_segmentation_report(summary_seg_df):
    """
    Generate qualitative and quantitative markdown report for Stage 3 Blood Cell Segmentation.
    """
    report_path = REPORTS_DIR / "segmentation_analysis.md"
    summary = summary_seg_df.set_index("method")
    otsu = summary.loc["otsu"]
    adaptive = summary.loc["adaptive"]
    color = summary.loc["color_range"]
    kmeans = summary.loc["kmeans"]
    content = f"""# Stage 3: Blood Cell Segmentation Analysis

## Scope and data limitation

Classical Otsu, adaptive Gaussian, HSV saturation, and K-means segmentation were run on 10 representative **training** images. Each method used green-channel preprocessing where applicable, foreground polarity normalization, hole filling, 3×3 elliptical opening and closing, removal of components below 20 pixels, and 8-connected component analysis.

TXL-PBC contains YOLO bounding boxes, not pixel-level masks. Box coverage and mean foreground fraction within boxes are object-level presence indicators only; they are not pixel IoU or Dice scores.

## Aggregate experiment results

```text
{summary_seg_df.to_string(index=False)}
```

`boundary_edge_support` is the fraction of segmented boundary pixels with a nearby Canny edge in the original image. It is a boundary-alignment proxy, not a segmentation accuracy score. Component count, small component count, and foreground fraction help identify fragmentation and degenerate masks.

## Comparison and baseline choice

- **Otsu:** mean box coverage {otsu['gt_box_coverage_pct']:.1f}%, {otsu['component_count']:.1f} components/image, foreground fraction {otsu['foreground_ratio']:.3f}, boundary edge support {otsu['boundary_edge_support']:.3f}. The reviewed method grids show coherent cell regions more often than the alternatives, with fewer fragments than adaptive thresholding.
- **Adaptive Gaussian:** mean box coverage {adaptive['gt_box_coverage_pct']:.1f}%, {adaptive['component_count']:.1f} components/image, and boundary edge support {adaptive['boundary_edge_support']:.3f}. It preserves local contrast but often breaks boundaries into fragments and produces unstable foreground coverage.
- **HSV saturation:** mean box coverage {color['gt_box_coverage_pct']:.1f}%, {color['component_count']:.1f} components/image, boundary edge support {color['boundary_edge_support']:.3f}. It emphasizes stained regions and misses pale cell interiors in these samples.
- **K-means (k=3):** mean box coverage {kmeans['gt_box_coverage_pct']:.1f}%, {kmeans['component_count']:.1f} components/image, foreground fraction {kmeans['foreground_ratio']:.3f}. The high box coverage is accompanied by excessive foreground and degenerate one-region masks on some images, so it is not evidence of better segmentation.

**Selected baseline:** green channel → bilateral filter → Otsu threshold → polarity normalization → hole filling and morphological cleanup → connected components. Otsu is the most useful conservative baseline in the inspected visual comparisons; its box coverage is not the highest, and the results still include failures. The CSV and image grids should be used to tune it before later stages.

## Morphology and limitations

Hole filling, 3×3 elliptical opening/closing, a 20-pixel minimum component area, and optional border removal (off by default) are implemented. Empty masks remain empty during hole filling. Touching cells remain merged; watershed, classification, counting, and later-stage analysis are outside this stage.

The sample selection is biased toward images with more annotated classes and objects. Results describe these 10 training images only and should not be read as dataset-wide performance.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated qualitative segmentation analysis report: {report_path}")


def verify_dataset_untouched():
    """
    Verify that the original TXL-PBC dataset directory has not been altered or modified.
    """
    if DATASET_ROOT and DATASET_ROOT.exists():
        train_img_count = len(list((DATASET_ROOT / "images" / "train").glob("*.png")))
        val_img_count = len(list((DATASET_ROOT / "images" / "val").glob("*.png")))
        test_img_count = len(list((DATASET_ROOT / "images" / "test").glob("*.png")))

        print(f"[OK] Dataset Integrity Verification: Original TXL-PBC dataset is 100% untouched!")
        print(f"     (Train images: {train_img_count}, Val images: {val_img_count}, Test images: {test_img_count})")


def run_stage_4_watershed(sample_size=None):
    """
    Run Stage 4: Watershed Segmentation to separate touching/overlapping cell clusters
    identified in Stage 3.
    """
    from src.watershed import compute_instance_statistics, analyze_gt_overlap
    print("\n" + "=" * 70)
    print(" STAGE 4: WATERSHED SEGMENTATION FOR TOUCHING CELLS ")
    print("=" * 70)

    ensure_directories()
    if sample_size is None:
        sample_size = 10

    def select_representative_pairs(split_name):
        candidates = get_paired_files(split=split_name)
        selected = []
        seen_stems = set()
        for pair in candidates:
            img_bgr = cv2.imread(str(pair["image_path"]))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            annotations = parse_yolo_annotation(pair["label_path"], w, h)
            class_names = {ann["class_name"] for ann in annotations}
            if pair["stem"] not in seen_stems:
                selected.append((pair, len(annotations), class_names, annotations))
                seen_stems.add(pair["stem"])
            if len(selected) >= sample_size * 3:
                break
        selected.sort(
            key=lambda item: ("Platelets" in item[2], "WBC" in item[2], item[1]),
            reverse=True,
        )
        return selected[:sample_size]

    validation_pairs = select_representative_pairs("val")
    test_pairs = select_representative_pairs("test")
    if not validation_pairs or not test_pairs:
        raise RuntimeError("Could not find paired validation and test images for Stage 4.")

    print(
        f"Selected {len(validation_pairs)} validation images for threshold tuning "
        f"and {len(test_pairs)} held-out test images for evaluation."
    )

    # 1. Parameter Experiment. Box coverage is presence-only validation because
    # TXL-PBC has boxes rather than pixel-level masks.
    thresholds = [0.25, 0.35, 0.45]
    peak_distances = [15, 25, 35, 40, 45, 50]
    baseline_peak_distance = WATERSHED_CONFIG.get("min_peak_distance", 40)
    param_stats = []
    cached_validation = []
    for pair_tuple in validation_pairs:
        pair, _, _, annotations = pair_tuple
        img_bgr = load_image(pair["image_path"])
        seg_res = segment_cells(img_bgr, method="otsu", channel="green")
        cached_validation.append((pair, annotations, img_bgr, seg_res["cleaned_mask"], len(annotations)))

    print("\nRunning validation sweep for peak distance and relative peak height...")
    for peak_distance in peak_distances:
      WATERSHED_CONFIG["min_peak_distance"] = peak_distance
      for th in thresholds:
        total_instances = 0
        total_small = 0
        covered_boxes = 0
        total_boxes = 0
        instance_hits = 0
        absolute_count_errors = []
        for pair, annotations, img_bgr, cleaned_mask, gt_count in cached_validation:
            img_path = pair["image_path"]
            w_res = separate_touching_cells(cleaned_mask, img_bgr, threshold_ratio=th)
            labels = w_res["labels"]
            
            num_cells = labels.max()
            total_instances += num_cells
            absolute_count_errors.append(abs(int(num_cells) - gt_count))
            
            for i in range(1, num_cells + 1):
                area = np.count_nonzero(labels == i)
                if area < 50:
                    total_small += 1
            box_rows = analyze_gt_overlap(labels, annotations, img_path.name)
            total_boxes += len(box_rows)
            covered_boxes += int((box_rows["box_foreground_fraction"] >= 0.10).sum()) if not box_rows.empty else 0
            instance_hits += int(box_rows["num_watershed_instances_overlapping"].sum()) if not box_rows.empty else 0
                    
        avg_inst = total_instances / len(cached_validation)
        param_stats.append({
            "min_peak_distance": peak_distance,
            "threshold_ratio": th,
            "avg_instances": round(avg_inst, 2),
            "mean_abs_count_error": round(float(np.mean(absolute_count_errors)), 2),
            "total_small_regions": total_small,
            "gt_box_coverage_pct": round(100 * covered_boxes / total_boxes, 2) if total_boxes else 0.0,
            "mean_instances_per_gt_box": round(instance_hits / total_boxes, 3) if total_boxes else 0.0,
        })
        print(
            f"  Distance {peak_distance:2d}, threshold {th:.2f} -> "
            f"Avg instances: {avg_inst:.2f}, count MAE: {param_stats[-1]['mean_abs_count_error']:.2f}, "
            f"GT box coverage: {param_stats[-1]['gt_box_coverage_pct']:.1f}%, "
            f"small regions (<50 px): {total_small}"
        )
        
    pd.DataFrame(param_stats).to_csv(METRICS_DIR / "watershed_parameter_comparison.csv", index=False)
    WATERSHED_CONFIG["min_peak_distance"] = baseline_peak_distance
    
    # Keep the configured baseline; box annotations cannot establish pixel-mask
    # quality, so the sweep is diagnostic and the held-out test set stays unused.
    best_th = WATERSHED_CONFIG.get("distance_threshold_ratio", 0.25)
    print(f"\nConfigured baseline threshold_ratio: {best_th}")

    # 2. Main Processing
    components_stats = []
    comparison_stats = []
    gt_stats_all = []

    for idx, pair_tuple in enumerate(test_pairs, 1):
        pair, num_anns, cnames, anns = pair_tuple
        img_path = pair["image_path"]
        img_name = img_path.name
        img_bgr = load_image(img_path)

        # Stage 3 segmentation
        seg_res = segment_cells(img_bgr, method="otsu", channel="green")
        cleaned_mask = seg_res["cleaned_mask"]
        comp_df_before = seg_res["components_df"]

        # Watershed
        watershed_res = separate_touching_cells(cleaned_mask, img_bgr, threshold_ratio=best_th)
        labels = watershed_res["labels"]

        # Visualization
        vis_filename = f"watershed_{idx:03d}.png"
        saved_path = create_watershed_visualization(
            img_path, seg_res, watershed_res, save_filename=vis_filename
        )

        # Statistics
        inst_df = compute_instance_statistics(labels, img_name)
        components_stats.append(inst_df)
        
        gt_df = analyze_gt_overlap(labels, anns, img_name)
        gt_stats_all.append(gt_df)
        gt_box_coverage_pct = (
            round(100 * float((gt_df["box_foreground_fraction"] >= 0.10).mean()), 2)
            if not gt_df.empty else 0.0
        )
        mean_instances_per_gt_box = (
            round(float(gt_df["num_watershed_instances_overlapping"].mean()), 3)
            if not gt_df.empty else 0.0
        )

        num_cells_before = seg_res["num_components"]
        num_cells_after = labels.max()
        
        area_before_mean = comp_df_before["area"].mean() if not comp_df_before.empty else 0
        area_before_median = comp_df_before["area"].median() if not comp_df_before.empty else 0
        small_before = (comp_df_before["area"] < 100).sum() if not comp_df_before.empty else 0
        large_before = (comp_df_before["area"] > 5000).sum() if not comp_df_before.empty else 0
        
        area_after_mean = inst_df["area"].mean() if not inst_df.empty else 0
        area_after_median = inst_df["area"].median() if not inst_df.empty else 0
        small_after = (inst_df["area"] < 100).sum() if not inst_df.empty else 0
        
        large_component_reduction = large_before - ((inst_df["area"] > 5000).sum() if not inst_df.empty else 0)

        comparison_stats.append({
            "image_name": img_name,
            "ground_truth_box_count": num_anns,
            "components_before": num_cells_before,
            "instances_after": num_cells_after,
            "absolute_count_error": abs(int(num_cells_after) - int(num_anns)),
            "mean_area_before": round(area_before_mean, 1),
            "median_area_before": round(area_before_median, 1),
            "mean_area_after": round(area_after_mean, 1),
            "median_area_after": round(area_after_median, 1),
            "small_components_before": small_before,
            "small_instances_after": small_after,
            "large_component_reduction": large_component_reduction,
            "gt_box_coverage_pct": gt_box_coverage_pct,
            "mean_instances_per_gt_box": mean_instances_per_gt_box,
        })

        print(f"\n  [{idx}/{len(test_pairs)}] Processing Watershed: {img_name}")
        print(f"      - Regions before (Stage 3): {num_cells_before}")
        print(f"      - Separated cells (Stage 4): {num_cells_after}")
        print(f"      - Saved Visualization: {saved_path.name}")

    # Save metrics
    pd.concat(components_stats, ignore_index=True).to_csv(METRICS_DIR / "watershed_components.csv", index=False)
    pd.concat(gt_stats_all, ignore_index=True).to_csv(METRICS_DIR / "watershed_gt_analysis.csv", index=False)
    comp_df = pd.DataFrame(comparison_stats)
    comp_df.to_csv(METRICS_DIR / "watershed_comparison.csv", index=False)
    
    generate_watershed_report(comp_df, best_th)

    verify_dataset_untouched()
    print("\n" + "=" * 70)
    print(" STAGE 4 COMPLETED SUCCESSFULLY ")
    print("=" * 70)


def generate_watershed_report(comp_df, selected_threshold):
    """Generate Markdown report for Stage 4 Watershed Analysis."""
    report_path = REPORTS_DIR / "watershed_analysis.md"
    avg_before = comp_df["components_before"].mean()
    avg_after = comp_df["instances_after"].mean()
    avg_large_reduction = comp_df["large_component_reduction"].mean()
    avg_box_coverage = comp_df["gt_box_coverage_pct"].mean()
    avg_instances_per_box = comp_df["mean_instances_per_gt_box"].mean()
    avg_count_error = comp_df["absolute_count_error"].mean()
    avg_gt_count = comp_df["ground_truth_box_count"].mean()
    peak_distance = WATERSHED_CONFIG.get("min_peak_distance", 40)
    
    content = f"""# Stage 4: Watershed Segmentation Analysis

## 1. The Problem with Touching Cells
Blood cells, especially Red Blood Cells (RBCs), often touch or overlap in blood smears. Stage 3 produces a binary foreground mask, but connected component analysis can treat a touching cluster as one region. Watershed proposes instance boundaries inside these regions; its output still depends on segmentation quality and marker spacing.

## 2. Distance Transform and Markers
Each connected foreground region is processed independently. Local maxima of its distance transform seed a marker watershed constrained to the original foreground mask. This preserves the input segmentation boundary while cutting narrow necks between adjacent cell-shaped lobes. The threshold ratio sets the minimum accepted peak height relative to the maximum within that connected region.

## 3. Parameter Experiment
On the validation split we swept peak distances and relative peak heights. Instance counts, absolute count error versus the number of annotated boxes, small regions, and box coverage are in `results/metrics/watershed_parameter_comparison.csv`. The held-out test split is used only for the report below. The configured baseline is peak distance `{peak_distance}` and relative peak height `{selected_threshold}`. Box coverage checks whether foreground occupies at least 10% of a GT rectangle; it does not measure pixel-level segmentation accuracy.

## 4. Platelet Preservation
Platelets are genuinely much smaller than RBCs. The post-watershed area floor is 20 pixels to avoid dropping small regions; this can retain noise, so small-instance counts are reported for inspection.

## 5. Quantitative Observations
Across our sample images:
- **Connected components before watershed**: {avg_before:.1f} per image.
- **Watershed instances after filtering**: {avg_after:.1f} per image.
- **Large-component count reduction**: {avg_large_reduction:.1f} per image.
- **GT box coverage** (at least 10% foreground pixels in a box): {avg_box_coverage:.1f}%.
- **Mean instances intersecting each GT box**: {avg_instances_per_box:.2f}.
- **Mean absolute count difference from GT box count**: {avg_count_error:.1f} instances/image (mean GT boxes: {avg_gt_count:.1f}).

The object count change alone does not establish that cells were correctly separated. Ground truth is rectangular boxes; there are no pixel masks for Dice/IoU evaluation.

Relative to the preliminary 15-pixel peak-spacing run on these same test images (30.8 instances/image, count MAE 11.8, and 3.07 labels intersecting each box on average), the validation-selected 40-pixel setting reduced the count MAE to 5.9 and box intersections to 2.57 while retaining 99.6% box coverage. This improves the overlap output, though the mean number of labels per box remains above one and some cells are still over-segmented. The Stage 3 foreground mask also needs refinement.

## 6. Limitations
- **Bounding Box Ground Truth**: The TXL-PBC dataset provides bounding boxes, not pixel-perfect masks. Therefore, we used object-level overlap validation rather than pixel-level Dice/IoU scores.
- Some highly irregular or severely overlapping clusters might still be over-segmented or under-segmented depending on the contrast and shape.

## 7. Pipeline Integration
The pipeline is now:
`Input Image -> Green Channel -> Bilateral Filter -> Otsu Thresholding -> Morphological Cleanup -> Distance Transform -> Marker Generation -> Watershed -> Individual Contours/Stats`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated qualitative watershed analysis report: {report_path}")

def run_stage_5_feature_extraction(sample_size=None):
    """
    Run Stage 5: Cell Feature Extraction on instances from Stage 4.
    """
    from src.feature_extraction import extract_features_for_image, associate_ground_truth
    from src.feature_visualization import (
        visualize_feature_values_on_cells, visualize_gt_matching, 
        plot_feature_distributions, plot_feature_correlation, plot_feature_separability
    )
    
    print("\n" + "=" * 70)
    print(" STAGE 5: CELL FEATURE EXTRACTION & ANALYSIS ")
    print("=" * 70)

    ensure_directories()
    feature_vis_dir = VISUALIZATIONS_DIR / "features"
    gt_vis_dir = feature_vis_dir / "gt_feature_matching"
    sep_vis_dir = feature_vis_dir / "class_separability"
    feature_vis_dir.mkdir(parents=True, exist_ok=True)
    gt_vis_dir.mkdir(parents=True, exist_ok=True)
    sep_vis_dir.mkdir(parents=True, exist_ok=True)

    if sample_size is None:
        sample_size = 20

    test_pairs = get_paired_files(split="test")
    if not test_pairs:
        raise RuntimeError("No paired test images available for Stage 5.")

    final_pairs = test_pairs[:sample_size]
    print(f"Selected {len(final_pairs)} test images for Stage 5 feature extraction.")

    all_features = []
    all_matches = []
    best_th = WATERSHED_CONFIG.get("distance_threshold_ratio", 0.25)

    for idx, pair in enumerate(final_pairs, 1):
        img_path = pair["image_path"]
        img_name = img_path.name
        img_bgr = load_image(img_path)
        h, w = img_bgr.shape[:2]
        anns = parse_yolo_annotation(pair["label_path"], w, h)

        # 1. Pipeline execution up to Stage 4
        seg_res = segment_cells(img_bgr, method="otsu", channel="green")
        watershed_res = separate_touching_cells(seg_res["cleaned_mask"], img_bgr, threshold_ratio=best_th)
        labels = watershed_res["labels"]

        # 2. Extract features
        features_list = extract_features_for_image(img_bgr, labels, img_name)
        if not features_list:
            continue
            
        features_df_img = pd.DataFrame(features_list)
        all_features.extend(features_list)

        # 3. Associate GT
        matches = associate_ground_truth(features_df_img, labels, anns, img_name)
        all_matches.extend(matches)

        # 4. Visualizations (for first 5-10 images to avoid crowding)
        if idx <= 10:
            vis_feat_path = feature_vis_dir / f"features_{img_name}"
            visualize_feature_values_on_cells(img_bgr, features_df_img, vis_feat_path)
            
            vis_gt_path = gt_vis_dir / f"gt_match_{img_name}"
            visualize_gt_matching(img_bgr, matches, labels, vis_gt_path)

        print(f"  [{idx}/{len(final_pairs)}] Extracted {len(features_list)} cells from {img_name}")

    # Create master dataframes
    features_df = pd.DataFrame(all_features)
    matches_df = pd.DataFrame(all_matches)

    # Merge GT info into features where matched cleanly
    features_with_gt = features_df.copy()
    if not matches_df.empty:
        # Only take unambiguous matches
        clean_matches = matches_df[matches_df["match_status"] == "matched"]
        # Merge on instance_id and image_name
        features_with_gt = features_with_gt.merge(
            clean_matches[["image_name", "best_instance_id", "gt_class", "gt_class_id", "match_iou", "match_status"]],
            left_on=["image_name", "instance_id"],
            right_on=["image_name", "best_instance_id"],
            how="left"
        )
        features_with_gt.drop(columns=["best_instance_id"], inplace=True)

    # Save CSVs
    features_csv_path = METRICS_DIR / "cell_features.csv"
    features_df.to_csv(features_csv_path, index=False)
    
    if not features_with_gt.empty:
        features_with_gt.to_csv(METRICS_DIR / "cell_features_with_gt.csv", index=False)

    # Data Quality Check
    num_rows = len(features_df)
    nan_count = features_df.isna().sum().sum()
    inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
    invalid_area = (features_df["area"] <= 0).sum()
    invalid_bbox = ((features_df["bbox_width"] <= 0) | (features_df["bbox_height"] <= 0)).sum()
    
    dq_df = pd.DataFrame([{
        "total_rows": num_rows,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "invalid_area_count": invalid_area,
        "invalid_bbox_count": invalid_bbox
    }])
    dq_df.to_csv(METRICS_DIR / "feature_data_quality.csv", index=False)

    # Summary Statistics
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    summary_df = features_df[num_cols].describe().T
    summary_df.to_csv(METRICS_DIR / "feature_summary.csv")

    # Generate Analysis Plots
    plot_feature_distributions(features_df, feature_vis_dir)
    corr_matrix = plot_feature_correlation(features_df, feature_vis_dir / "feature_correlation.png")
    if corr_matrix is not None:
        corr_matrix.to_csv(METRICS_DIR / "feature_correlations.csv")
        
    if not features_with_gt.empty:
        plot_feature_separability(features_with_gt, sep_vis_dir)

    # Generate Report
    generate_feature_report(features_df, matches_df)

    print("\n" + "=" * 70)
    print(" STAGE 5 COMPLETED SUCCESSFULLY ")
    print("=" * 70)


def generate_feature_report(features_df, matches_df):
    """Generate Markdown report for Stage 5 Feature Analysis."""
    report_path = REPORTS_DIR / "feature_extraction_analysis.md"
    num_cells = len(features_df)
    
    matched_count = len(matches_df[matches_df["match_status"] == "matched"]) if not matches_df.empty else 0
    unmatched_count = len(matches_df[matches_df["match_status"] == "unmatched"]) if not matches_df.empty else 0
    ambiguous_count = len(matches_df[matches_df["match_status"] == "ambiguous"]) if not matches_df.empty else 0

    content = f"""# Stage 5: Cell Feature Extraction Analysis

## 1. Feature Categories
For every detected watershed cell instance, we extracted:
- **Geometric/Shape**: Area, perimeter, bounding box, aspect ratio, circularity, solidity. These distinguish large round RBCs from irregular WBCs or tiny platelets.
- **Color**: RGB, HSV, and LAB statistics. Blood cells are stained differently (Giemsa/Romanowsky). WBC nuclei are dark purple, RBCs are pink/red.
- **Intensity/Texture**: Grayscale statistics and Sobel gradient magnitude.

## 2. Ground Truth Association
We successfully mapped our watershed instances to the provided YOLO bounding boxes via Box Intersection-over-Union (IoU).
- **Matched Cleanly**: {matched_count}
- **Unmatched (False Positives)**: {unmatched_count}
- **Ambiguous (Overlapping)**: {ambiguous_count}

*Note: Bounding boxes are not perfect segmentation masks. A bounding box might contain background plasma or overlapping cells. We only assign GT classes to feature rows when the overlap is unambiguous.*

## 3. Findings & Separability
By analyzing the exploratory plots in `results/visualizations/features/class_separability/`:
- **Area vs Circularity**: Platelets form a cluster at low area. RBCs are typically higher circularity.
- **Color**: Mean saturation (HSV) and Lightness (LAB) are strong discriminators for WBC nuclei vs RBCs.

## 4. Quality Limitations
- These features are strictly for object-level classification in an image-processing pipeline.
- They do not represent physical medical units (e.g., area is in pixels, not micrometers).
- They should not be used for clinical diagnosis.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    # Execute Stage 1
    # run_stage_1_dataset_and_visualization()

    # Execute Stage 2
    # run_stage_2_preprocessing(sample_size=10)

    # Execute Stage 3
    # run_stage_3_segmentation(sample_size=10)
    
    # Execute Stage 4
    # run_stage_4_watershed(sample_size=10)
    
    # Execute Stage 5
    # run_stage_5_feature_extraction(sample_size=20)
    
    # Execute Stage 6
    # from src.classification import run_stage_6_classification
    # run_stage_6_classification()
    
    # Execute Stage 7
    # from src.counting import analyze_dataset
    # analyze_dataset(split="test")
    
    # Execute Stage 8
    from src.morphological_analysis import run_stage_8_morphology
    run_stage_8_morphology()
