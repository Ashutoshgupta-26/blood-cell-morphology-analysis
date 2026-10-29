# 🩸 Blood Cell Detection, Counting & Morphological Analysis
### An End-to-End Classical Image Processing & Machine Learning Pipeline

> **⚠️ Medical Disclaimer:** This project is strictly an educational and research image-analysis tool. It is **not** a clinical diagnostic system. Results must **not** be used for medical decisions. All measurements are image-based proxies and do not equate to clinical laboratory blood counts.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: TXL-PBC](#2-dataset-txl-pbc)
3. [System Architecture & Pipeline](#3-system-architecture--pipeline)
4. [Stage 1 — Dataset Inspection & Ground-Truth Visualization](#4-stage-1--dataset-inspection--ground-truth-visualization)
5. [Stage 2 — Image Preprocessing & Color Space Analysis](#5-stage-2--image-preprocessing--color-space-analysis)
6. [Stage 3 — Blood Cell Segmentation](#6-stage-3--blood-cell-segmentation)
7. [Stage 4 — Watershed Segmentation (Touching Cell Separation)](#7-stage-4--watershed-segmentation-touching-cell-separation)
8. [Stage 5 — Cell Feature Extraction](#8-stage-5--cell-feature-extraction)
9. [Stage 6 — Cell Classification (Classical ML)](#9-stage-6--cell-classification-classical-ml)
10. [Stage 7 — Automated Cell Counting & Statistical Analysis](#10-stage-7--automated-cell-counting--statistical-analysis)
11. [Stage 8 — Morphological Analysis & Outlier Detection](#11-stage-8--morphological-analysis--outlier-detection)
12. [Streamlit Web Application](#12-streamlit-web-application)
13. [Evaluation Metrics & Results Summary](#13-evaluation-metrics--results-summary)
14. [Directory Structure](#14-directory-structure)
15. [Installation & Reproducibility](#15-installation--reproducibility)
16. [Key Limitations](#16-key-limitations)
17. [Technology Stack](#17-technology-stack)

---

## 1. Project Overview

This project implements a fully automated, **classical** (non-deep-learning) computer vision pipeline for the analysis of peripheral blood smear microscopy images. The pipeline detects, segments, classifies, and counts three major blood cell types:

| Cell Type | Abbreviation | Biological Role |
|---|---|---|
| Red Blood Cells | **RBC** | Oxygen transport via hemoglobin |
| White Blood Cells | **WBC** | Immune response and defense |
| Platelets | **PLT** | Hemostasis and blood clotting |

**Key Design Principle:** The entire pipeline relies on classical image processing (OpenCV, scikit-image) and classical machine learning (scikit-learn). No convolutional neural networks or deep learning object detectors are used, making the system fully interpretable and lightweight.

---

## 2. Dataset: TXL-PBC

The pipeline is trained and evaluated on the **TXL-PBC (Tianjin University Lab – Peripheral Blood Cell)** dataset.

| Property | Value |
|---|---|
| Total Images | 1,260 |
| Image Format | PNG |
| Annotation Format | YOLO Bounding Boxes |
| Classes | RBC (ID=1), WBC (ID=0), Platelets (ID=2) |
| Train Split | 882 images |
| Validation Split | 252 images |
| Test Split | 126 images |
| Staining Method | Romanowsky / Giemsa |

**Annotation Limitation:** The TXL-PBC dataset provides *bounding boxes*, not pixel-level segmentation masks. This means pixel IoU / Dice scores cannot be computed. All segmentation quality is assessed via object-level *box coverage* metrics (presence-based overlap).

**Dataset Integrity Policy:** The pipeline never modifies the original dataset. A post-stage verification step (`verify_dataset_untouched()`) confirms all original image and label counts are unchanged after every pipeline run.

---

## 3. System Architecture & Pipeline

The complete pipeline is an eight-stage sequential system, where each stage's output feeds the next:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Microscopy Image (PNG)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────┐
         │  STAGE 1: Dataset Inspection &       │
         │           GT Visualization           │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 2: Image Preprocessing &      │
         │           Color Space Analysis       │
         │  (Green Channel → Bilateral → CLAHE) │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 3: Binary Segmentation        │
         │  (Otsu Thresholding → Morphological  │
         │   Cleanup → Connected Components)    │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 4: Watershed Segmentation     │
         │  (Distance Transform → Local Maxima  │
         │   → Watershed → Individual Instances)│
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 5: Feature Extraction         │
         │  (32 Geometric + Color + Texture     │
         │   Features per Cell Instance)        │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 6: Cell Classification (ML)   │
         │  (SVM / Random Forest / KNN / LR)    │
         │  → RBC | WBC | Platelets             │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 7: Cell Counting &            │
         │           Statistical Analysis       │
         │  (Per-Image Counts + Error Metrics)  │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  STAGE 8: Morphological Analysis &   │
         │           Outlier Detection          │
         │  (Z-Score, IQR, Isolation Forest)    │
         └─────────────────┬───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              OUTPUT: Reports, Visualizations, Metrics               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1 — Dataset Inspection & Ground-Truth Visualization

**Module:** `main.py → run_stage_1_dataset_and_visualization()`  
**Source:** `src/dataset_utils.py`, `src/annotation_utils.py`, `src/visualization.py`

### Purpose
Validates the dataset is correctly paired (every image has a corresponding `.txt` label file), computes global statistics, and produces annotated visualizations to confirm label quality.

### Key Operations

1. **Dataset Pairing Verification** — Scans all three splits (train/val/test) and pairs each image with its YOLO annotation file. Unpaired files are flagged.

2. **Statistics Reporting** — Reports total counts per split and class distribution.

3. **Ground-Truth Bounding Box Drawing** — Loads 3 representative training images, parses their YOLO annotations, and overlays color-coded bounding boxes:
   - 🟣 **WBC** — Purple/Magenta
   - 🔴 **RBC** — Red
   - 🟡 **Platelets** — Yellow/Orange

4. **Side-by-Side Comparison Figures** — Saves the original image alongside the annotated version for visual inspection.

### YOLO Annotation Parsing
Annotations are stored in normalized YOLO format: `class_id x_center y_center width height`. The parser de-normalizes coordinates back to pixel space using the actual image dimensions.

### Outputs
```
results/visualizations/
    annotated_image_001.png
    annotated_image_002.png
    side_by_side_001.png
    side_by_side_002.png
```

---

## 5. Stage 2 — Image Preprocessing & Color Space Analysis

**Module:** `main.py → run_stage_2_preprocessing()`  
**Source:** `src/preprocessing.py`

### Purpose
Systematically evaluates multiple preprocessing strategies to identify the optimal pipeline for preparing blood smear images for segmentation.

### Color Space Analysis
Blood smear images are acquired in BGR. The pipeline investigates several color representations:

| Representation | Rationale |
|---|---|
| **Green Channel** | Hemoglobin absorbs green light most strongly. Produces highest contrast between red cells and pale background. |
| **Standard Grayscale** | Weighted average (0.299R + 0.587G + 0.114B). Good general purpose. |
| **HSV – Saturation** | Highlights stained regions but misses pale interiors. |
| **LAB – L Channel** | Perceptual lightness; useful for illumination normalization. |

**Selected:** Green Channel — empirically verified to give the best foreground/background contrast for Romanowsky-stained slides.

### Noise Reduction Filters Compared

| Filter | Parameters | Behavior |
|---|---|---|
| **Gaussian Blur** | Kernel 5×5, σ=1.0 | Uniform smoothing; blurs edges |
| **Median Blur** | Kernel 5×5 | Salt-and-pepper noise removal; preserves edges better |
| **Bilateral Filter** | d=9, σColor=75, σSpace=75 | **Edge-preserving** smoothing; best for cell boundary preservation |

**Selected:** Bilateral Filter — preserves sharp cell edges while effectively reducing photon noise and stain artifacts.

### Contrast Enhancement Methods

| Method | Description |
|---|---|
| **Global Histogram Equalization** | Redistributes intensity globally. Overenhances some regions. |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization with clipLimit=2.0, tileGrid=8×8. Applies local equalization to prevent over-amplification. **Selected.** |
| **Illumination Correction** | Background estimation using large Gaussian blur (51×51) + CLAHE |

### Quantitative Metrics Computed Per Image

For each preprocessing variant, the following statistics are computed and exported to `preprocessing_statistics.csv`:

- **Mean**: Average pixel intensity
- **Standard Deviation (Std)**: Measure of contrast
- **RMS Contrast** (`contrast_rms`): $\sqrt{E[I^2] - E[I]^2}$
- **Shannon Entropy** (`entropy`): Measure of image information content

### Recommended Pipeline (Output of Stage 2)

```
Raw Image (BGR)
    ↓
Green Channel Extraction
    ↓
Bilateral Filter (d=9, σColor=75, σSpace=75)   ← Edge-preserving smoothing
    ↓
CLAHE Enhancement (clipLimit=2.0, tileGrid=8×8) ← Local contrast boost
    ↓
Clean, High-Contrast Image → Stage 3 Segmentation
```

### Outputs
```
results/
    visualizations/preprocessing/
        preprocessing_comparison_001.png  (9-panel grid)
        color_space_comparison_001.png
    metrics/
        preprocessing_statistics.csv
    reports/
        preprocessing_analysis.md
```

---

## 6. Stage 3 — Blood Cell Segmentation

**Module:** `main.py → run_stage_3_segmentation()`  
**Source:** `src/segmentation.py`

### Purpose
Converts the preprocessed grayscale image into a binary foreground mask — separating cell bodies (foreground) from background plasma.

### Segmentation Methods Evaluated

| Method | Description | Notes |
|---|---|---|
| **Otsu Thresholding** | Automatically finds the optimal global threshold by minimizing intra-class variance | **Selected baseline** |
| **Adaptive Gaussian** | Local threshold computed per block (size=21, C=5.0) | More fragments, unstable |
| **HSV Saturation** | Thresholds on HSV Saturation channel (range 30–255) | Misses pale cell interiors |
| **K-Means (k=3)** | Clusters pixels into 3 intensity groups | Excessive foreground in some images |

### Otsu's Method — Technical Detail

Otsu's algorithm selects a threshold **T*** that minimizes the weighted sum of intra-class variances:

```
σ²_within(T) = w₀(T) · σ²₀(T) + w₁(T) · σ²₁(T)

Where:
  w₀, w₁ = class probabilities (background and foreground)
  σ²₀, σ²₁ = class variances
```

This is equivalent to maximizing inter-class variance, ensuring maximum separation between cell pixels and background.

### Polarity Normalization
After thresholding, a polarity check ensures cells are white (255) and background is black (0). If the foreground region is accidentally inverted (e.g., very bright background slide), the mask is flipped.

### Morphological Cleanup Pipeline

After initial thresholding, the binary mask goes through a structured cleanup:

```
Thresholded Binary Mask
    ↓
Hole Filling (ndimage binary_fill_holes)   ← Fill hollow cell interiors
    ↓
Morphological Opening (3×3 Elliptical Kernel) ← Remove thin protrusions/noise
    ↓
Morphological Closing (3×3 Elliptical Kernel) ← Bridge small gaps
    ↓
Small Component Removal (area < 100 px)    ← Eliminate noise specks
    ↓
Clean Binary Mask
```

**Minimum Component Area (100 px):** Set to retain even small platelets (empirically, minimum platelet area ≈ 133 px) while removing noise artifacts.

**Border Removal:** Disabled by default to avoid deleting real cells that touch the image edge.

### Connected Component Analysis
After cleanup, `cv2.connectedComponentsWithStats` (8-connectivity) labels each contiguous foreground region as a distinct "component". Each component includes:
- Bounding box (x, y, width, height)
- Pixel area
- Centroid coordinates

### Evaluation Metric: GT Box Coverage
Since pixel masks are unavailable, a proxy metric computes what fraction of ground-truth bounding boxes have at least some foreground pixels inside them:
```
Coverage % = (boxes with fg_ratio > threshold) / total_gt_boxes × 100
```

### Segmentation Method Comparison (Averaged across 10 images)
Comparative metrics are exported to `results/metrics/segmentation_method_comparison.csv`.

Key comparative metrics:
- **Foreground Ratio** — fraction of image pixels classified as foreground
- **Component Count** — number of detected instances
- **Mean Component Area** — average detected cell size
- **Boundary Edge Support** — fraction of segmentation boundaries aligned with Canny edges
- **GT Box Coverage %** — object-level detection presence proxy

### Outputs
```
results/
    visualizations/segmentation/
        segmentation_comparison_001.png  (6-panel pipeline)
        segmentation_methods_001.png     (4-panel method comparison)
    metrics/
        segmentation_method_comparison.csv
        segmentation_components.csv
    reports/
        segmentation_analysis.md
```

---

## 7. Stage 4 — Watershed Segmentation (Touching Cell Separation)

**Module:** `main.py → run_stage_4_watershed()`  
**Source:** `src/watershed.py`

### The Problem: Touching Cells
Blood smear images frequently contain clusters of RBCs pressed against each other. Stage 3's connected component analysis treats an entire touching cluster as **one** component. Watershed splits these clusters into individual cells.

### Algorithm: Distance Transform + Marker Watershed

For each connected foreground region independently:

```
Connected Component Mask
    ↓
Distance Transform                     ← Each pixel = distance to nearest background
    ↓
Normalize Distance Map (0.0 to 1.0)
    ↓
Find Local Maxima                      ← Peak = center of each individual cell
  (min_peak_distance = 40 px,
   min_peak_height = threshold_ratio × max_distance)
    ↓
Create Integer Marker Map              ← Each peak = unique integer seed
    ↓
Scipy Watershed (inverted distance)    ← Fills basins from seeds, bounded by foreground mask
    ↓
Filter tiny instances (area < 20 px)   ← Retain platelets, remove artifacts
    ↓
Individual Cell Labels
```

**Key Parameter — `min_peak_distance` (40 px):** Controls minimum spacing between cell centers. Prevents over-splitting by requiring detected peaks to be at least this far apart. Scaled proportionally to image short-edge.

**Key Parameter — `distance_threshold_ratio` (0.35):** A peak must reach at least 35% of the maximum distance value within its connected region to be accepted as a real cell center marker.

### Parameter Sweep (Validation Tuning)

A systematic grid search is run on the **validation split** over:
- `min_peak_distance` ∈ {15, 25, 35, 40, 45, 50}
- `distance_threshold_ratio` ∈ {0.25, 0.35, 0.45}

Metrics tracked per configuration:
- Average instance count per image
- Mean Absolute Count Error (MAE vs. GT box count)
- Small region count (< 50 px) — proxy for noise/over-segmentation
- GT Box Coverage % — proxy for detection recall

**Final Selected Configuration:** `min_peak_distance=40`, `threshold_ratio=0.35`

This configuration reduced count MAE from ~11.8 (naive 15 px spacing) to ~5.9 while retaining 99.6% GT box coverage.

### Platelet Preservation
The post-watershed minimum area is intentionally set to **20 pixels** (not 100) to ensure small-but-valid platelet instances are preserved.

### Quantitative Results (10 Test Images)

| Metric | Value |
|---|---|
| Average connected components before watershed | ~N (varies) |
| Average watershed instances after filtering | Increases by separating clusters |
| Mean absolute count error vs GT | ~5.9 per image |
| GT box coverage (≥10% foreground in box) | ~99.6% |

### Outputs
```
results/
    visualizations/watershed/
        watershed_001.png   (4-panel: original, mask, distance transform, instances)
    metrics/
        watershed_parameter_comparison.csv
        watershed_components.csv
        watershed_gt_analysis.csv
        watershed_comparison.csv
    reports/
        watershed_analysis.md
```

---

## 8. Stage 5 — Cell Feature Extraction

**Module:** `main.py → run_stage_5_feature_extraction()`  
**Source:** `src/feature_extraction.py`

### Purpose
For every individual watershed cell instance, extract a comprehensive numeric feature vector that captures its geometry, color, intensity, and texture.

### Feature Engineering (32 Features per Cell)

#### 🔷 Group 1: Geometric / Shape Features

| Feature | Formula / Source | Biological Significance |
|---|---|---|
| `area` | Pixel count of instance mask | Size discriminator (WBC >> RBC >> PLT) |
| `perimeter` | Contour perimeter | Boundary complexity |
| `equivalent_diameter` | $\sqrt{4 \cdot area / \pi}$ | Effective circle diameter |
| `aspect_ratio` | bbox_width / bbox_height | Shape elongation |
| `extent` | area / (bbox_w × bbox_h) | Fill factor within bounding box |
| `circularity` | $4\pi \cdot area / perimeter^2$ | 1.0 = perfect circle; RBCs are highly circular |
| `convex_hull_area` | Area of convex hull | Used to compute solidity |
| `solidity` | area / convex_hull_area | Compactness; WBCs are irregular |
| `orientation` | Major axis angle | Rotation angle of the cell |
| `enclosing_circle_radius` | Min enclosing circle radius | Gross size estimate |
| `bbox_width`, `bbox_height` | Bounding box dimensions | Spatial extent |
| `centroid_x`, `centroid_y` | Centroid coordinates | Position |
| `normalized_centroid_*` | Centroid / image dimension | Position-invariant |

#### 🔶 Group 2: Color Features

Color statistics are computed inside each cell's pixel mask (not the bounding box):

| Feature Set | Channels | Notes |
|---|---|---|
| **RGB** | `mean_r`, `mean_g`, `mean_b`, `std_r`, `std_g`, `std_b` | Raw color channels |
| **HSV** | `mean_h`, `mean_s`, `mean_v`, `std_h`, `std_s`, `std_v` | Hue-Saturation; WBC nuclei dark purple → high saturation |
| **LAB** | `mean_l`, `mean_a`, `mean_lab_b`, `std_l`, `std_a`, `std_lab_b` | Perceptual space; L channel = lightness |

**Key Discriminator — `mean_s` (HSV Saturation):** WBC nuclei are highly stained (dark purple), giving high saturation values. RBCs are pink/red with moderate saturation.

#### 🟢 Group 3: Intensity Features

| Feature | Description |
|---|---|
| `mean_intensity` | Mean grayscale value within mask |
| `median_intensity` | Median grayscale value |
| `std_intensity` | Standard deviation of grayscale |
| `min_intensity` / `max_intensity` | Intensity range |
| `intensity_p25` / `intensity_p75` | Quartile values |

#### 🔴 Group 4: Texture Features (Sobel Gradient)

| Feature | Description |
|---|---|
| `mean_gradient` | Mean Sobel edge magnitude within cell | Captures internal texture variation |
| `std_gradient` | Std of Sobel edge magnitude | WBCs have rough lobed nuclei → higher gradient |

**Gradient Computation:**
```python
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = sqrt(sobelx² + sobely²)
```

### Ground Truth Association (IoU-Based Matching)

Each extracted cell instance is matched to the closest GT bounding box using Intersection-over-Union:

```
IoU = intersection_area / union_area

Match Rules:
  IoU ≥ 0.3  AND  unique best overlap → "matched"
  IoU ≥ 0.3  AND  multiple overlaps, best < 0.6 → "ambiguous"  
  IoU < 0.3  → "unmatched"
```

Only **clean matches** are used to build the labeled classification dataset, preventing mislabeling from over/under-segmented cells.

### Data Quality Checks
After extraction, the pipeline verifies:
- Total NaN count across all feature columns
- Total Inf count across numeric columns  
- Invalid area count (area ≤ 0)
- Invalid bounding box count

### Outputs
```
results/
    metrics/
        cell_features.csv              (all features, no GT label)
        cell_features_with_gt.csv      (features + matched GT class)
        feature_data_quality.csv
        feature_summary.csv
        feature_correlations.csv
    visualizations/features/
        features_<image>.png           (feature overlay visualization)
        gt_feature_matching/
        class_separability/
    reports/
        feature_extraction_analysis.md
```

---

## 9. Stage 6 — Cell Classification (Classical ML)

**Module:** `src/classification.py → run_stage_6_classification()`

### Purpose
Train and compare classical machine learning classifiers to assign RBC / WBC / Platelets labels to individual watershed cell instances.

### Data Split Strategy (Image-Level)
To prevent **data leakage**, splitting is performed at the **image level** — not at the cell level. Cells from the same image never appear in both training and test sets:

```
Train Split Images → Extract cells → X_train
Val Split Images   → Extract cells → X_val  
Test Split Images  → Extract cells → X_test
```

This mirrors real-world deployment where the model must generalize to completely unseen images.

A formal **data leakage check** is run to confirm zero image overlap across splits and exported to `data_leakage_check.txt`.

### Feature Sets

**Feature Set A (Core, 11 features):**
`area`, `equivalent_diameter`, `aspect_ratio`, `circularity`, `solidity`, `mean_r`, `mean_g`, `mean_b`, `mean_s`, `mean_intensity`, `mean_gradient`

**Feature Set B (Full, 32 features) — Used for primary experiments:**
Complete geometric, RGB, HSV, LAB, intensity, and texture features.

### Preprocessing: Standard Scaling
All features are scaled using `StandardScaler` (zero mean, unit variance) fitted **only on training data** and applied to validation and test sets. The fitted scaler is saved as `results/models/scaler.joblib`.

### Classifiers Evaluated

| Classifier | Configuration | Notes |
|---|---|---|
| **Rule Baseline** | Area < 500 px → PLT; mean_s > 100 → WBC; else → RBC | Simple heuristic |
| **KNN** | K=5, Euclidean distance | Non-parametric |
| **SVM (RBF)** | C=1.0, gamma='scale', class_weight='balanced' | **Best classifier** |
| **Random Forest** | 100 estimators, class_weight='balanced' | Ensemble; interpretable |
| **Logistic Regression** | max_iter=1000, class_weight='balanced' | Linear baseline |

### Class Imbalance Handling
RBCs dominate in every blood smear image (typically 80–90% of all cells). All classifiers use `class_weight='balanced'` to up-weight minority classes (WBC, Platelets) during training. **Macro F1** (unweighted average across all classes) is the primary evaluation metric.

### Evaluation Metrics

For each classifier:
- **Accuracy** — overall correct classifications
- **Macro Precision / Recall / F1** — unweighted class average (primary for imbalanced data)
- **Weighted Precision / Recall / F1** — frequency-weighted class average
- **Confusion Matrix** — visualized as a heatmap

### Feature Ablation Study (Random Forest)

To assess which feature groups contribute most, Random Forest is retrained using subsets:

| Feature Group | Features Included |
|---|---|
| Shape | area, perimeter, equivalent_diameter, aspect_ratio, circularity, solidity |
| Color | mean_r, mean_g, mean_b, mean_h, mean_s, mean_v |
| Intensity | mean_intensity, std_intensity |
| Texture | mean_gradient, std_gradient |
| Shape + Color | area, circularity, mean_s, mean_r |
| All (Set B) | All 32 features |

**Key Finding:** Shape alone cannot distinguish all classes (especially stained WBCs from large RBCs). **Color (especially HSV Saturation)** is the decisive discriminator.

### Feature Importance (Random Forest)
Top discriminating features (by mean decrease in impurity):
1. `area` / `equivalent_diameter` — primary size discriminator
2. `mean_s` (HSV Saturation) — WBC vs. RBC key signal
3. `mean_intensity` — overall brightness
4. `circularity` — shape regularity

### Model Persistence
Trained models are saved for inference in Stage 7:
```
results/models/
    scaler.joblib            ← StandardScaler fitted on training data
    best_svm.joblib          ← Trained SVM model
    best_knn.joblib          ← Trained KNN model
    best_random_forest.joblib
    best_model_features.json ← Feature schema list
```

### Results

- **SVM (RBF) Macro F1:** ~1.000 on cleanly isolated single-cell crops
- **Random Forest Macro F1:** ~1.000 on cleanly isolated single-cell crops

> **Note:** Perfect scores reflect the fact that well-isolated individual cells are highly separable by area + color features. The harder problem is Stage 4 (correctly isolating cells), not Stage 6 (classifying them once isolated).

### Outputs
```
results/
    metrics/
        classification_dataset.csv
        class_distribution.csv
        classifier_comparison.csv
        feature_ablation.csv
        data_leakage_check.txt
        classification_report_svm.txt
        classification_errors.csv
    visualizations/classification/
        class_distribution.png
        classifier_comparison.png
        feature_ablation.png
        confusion_matrix_svm.png
        confusion_matrix_random_forest.png
        random_forest_feature_importance.png
        errors/                          (error crop images)
    reports/
        classification_analysis.md
```

---

## 10. Stage 7 — Automated Cell Counting & Statistical Analysis

**Module:** `src/counting.py → analyze_dataset(split='test')`

### Purpose
Run the complete end-to-end inference pipeline on every image in a dataset split, produce per-image cell counts (RBC, WBC, PLT, Total), compare against ground truth, and generate detection precision/recall metrics.

### End-to-End Inference Per Image

```
Input Image (BGR)
    ↓ load_image()
Preprocessing (Green → Bilateral → CLAHE)
    ↓ segment_cells(method='otsu', channel='green')
Binary Mask + Morphological Cleanup
    ↓ separate_touching_cells(threshold_ratio=0.35)
Watershed Instance Labels
    ↓ extract_features_for_image()
32-Feature DataFrame (one row per cell)
    ↓ scaler.transform() → model.predict()
Predicted Class per Instance
    ↓ Post-Classification Quality Filter
Filtered Predictions
    ↓ value_counts()
Per-Image Counts {RBC, WBC, Platelets, Total}
```

### Post-Classification Quality Filter

After classification, a rule-based filter removes obvious false positives without affecting true cells:

1. **Rule 1 — Small RBC Fragments (Interior):** RBC predictions with `area < 300 px` that do not touch the image border are removed. These are typically watershed over-segmentation artifacts.

2. **Rule 2 — Irregular RBC Fragments (Interior):** RBC predictions with `circularity < 0.25` AND `solidity < 0.6` that do not touch the border are removed. Real RBCs are nearly circular.

3. **Border Exception:** Both rules are waived for instances touching the image boundary (within 3 px of any edge), as these may be genuine cells partially cropped by the camera field of view.

### Cell Density Metrics

In addition to raw counts, cell density is computed:
```
cells_per_pixel = Total / (width × height)
cells_per_10000_pixels = cells_per_pixel × 10,000
cells_per_megapixel = cells_per_pixel × 1,000,000
```

### Count Error Metrics (Image-Level)

Comparing predicted counts to GT bounding box counts per image:

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | $\frac{1}{N}\sum|pred - gt|$ | Average absolute count error |
| **RMSE** | $\sqrt{\frac{1}{N}\sum(pred-gt)^2}$ | Penalizes large errors more |
| **MSE_signed** | $\frac{1}{N}\sum(pred - gt)$ | Negative = systematic undercounting |
| **MAPE** | $\frac{1}{N}\sum\frac{|pred-gt|}{gt}\times100$ | Percentage-relative error |

### Object Detection Metrics (IoU ≥ 0.5)

Using strict 0.5 IoU threshold for GT box ↔ prediction matching:

```
True Positive (TP):  Prediction overlaps GT box (IoU ≥ 0.5) AND correct class
False Positive (FP): Prediction with no GT match, OR correct overlap but wrong class
False Negative (FN): GT box with no matching prediction
```

Precision, Recall, and F1 computed per class and overall.

### Annotated Visualization Outputs

For each processed image, two annotated versions are saved:

1. **Boundary Image** — Contours drawn around each detected cell in class color, with ID labels and edge-cell annotations `(edge)`.
2. **Segmentation Overlay** — Translucent colored fill (α=0.4) over each cell, with count summary box.

### Results Summary

| Metric | Value |
|---|---|
| Total Counting MAE | 10.80 cells/image |
| Total Counting RMSE | 12.71 |
| Overall Object Detection Precision | (see report) |
| Overall Object Detection Recall | **~16%** (bottlenecked by watershed) |
| Classification Accuracy (on matched cells) | Very high |

> **Critical Finding:** The low recall (~16%) is almost entirely caused by **watershed under-segmentation** — dense touching RBC clusters remain as single components, making the entire cluster count as 1 detection instead of 20+. Classification accuracy on properly isolated cells is near-perfect.

### Outputs
```
results/
    metrics/
        cell_counts.csv
        ground_truth_counts.csv
        cell_predictions.csv
        count_comparison.csv
        counting_metrics.csv
        counting_summary.csv
        counting_error_ranking.csv
        object_detection_metrics.csv
        matched_classification_results.csv
    visualizations/counting/
        boundaries/               (contour-annotated images)
        segmentation_overlays/    (filled colored overlays)
        best_cases/
        worst_cases/
        count_distributions/
        rbc_predicted_vs_gt.png
        wbc_predicted_vs_gt.png
        platelet_predicted_vs_gt.png
        error_distributions.png
    reports/
        counting_analysis.md
```

---

## 11. Stage 8 — Morphological Analysis & Outlier Detection

**Module:** `src/morphological_analysis.py → run_stage_8_morphology()`

### Purpose
Analyze the size, shape, color, intensity, and texture distribution of all detected blood cell instances. Identify cells whose image-derived characteristics are statistically unusual within their predicted class — **image-based morphological outliers**.

### Feature Groups Analyzed

| Group | Features |
|---|---|
| **SIZE** | area, equivalent_diameter, perimeter, bbox_width, bbox_height |
| **SHAPE** | circularity, aspect_ratio, extent, solidity, convex_hull_area |
| **COLOR** | mean_r, mean_g, mean_b, mean_h, mean_s, mean_v, mean_l, mean_a, mean_lab_b |
| **INTENSITY** | mean_intensity, median_intensity, std_intensity, intensity_p25, intensity_p75 |
| **TEXTURE** | mean_gradient, std_gradient |

### Descriptive Statistics Per Class
For each class (RBC, WBC, Platelets) and each feature, the following are computed:
count, mean, median, std, min, Q25, Q75, max.

Exported to `morphology_class_statistics.csv`.

### Outlier Detection — Three Independent Methods

Outlier detection runs **within each class separately** to avoid comparing platelets (small) to WBCs (large).

#### Method 1: Univariate Z-Score
```
Z = (x - μ) / σ
Flagged if |Z| > 3.0
```
Identifies cells whose single-feature value is more than 3 standard deviations from their class mean.

#### Method 2: Univariate IQR Bounds
```
Lower bound = Q25 - 1.5 × IQR
Upper bound = Q75 + 1.5 × IQR
Flagged if x < Lower OR x > Upper
```
Robust to non-Gaussian distributions; less sensitive to extreme outliers compared to Z-score.

#### Method 3: Multivariate Isolation Forest
```
Contamination = 0.05 (expects ~5% outliers)
Features: all valid numeric features (no NaN), StandardScaler applied
```
Isolation Forest assigns an anomaly score to each cell based on how easily it can be "isolated" in the feature space. Cells that are isolated in very few splits are flagged as outliers.

Applied only when a class has ≥ 20 instances (minimum for reliable statistics).

### High-Confidence Outlier Definition
```
outlier_score = is_z_outlier + is_iqr_outlier + is_multivariate_outlier

high_confidence_morphological_outlier = (outlier_score >= 2)
```
A cell must be flagged by **at least 2 out of 3 independent methods** to be classified as a high-confidence morphological outlier.

### Visualizations Generated
- **Boxplots** of size and shape features by class
- **Scatter plots** of shape relationships (area vs. circularity, diameter vs. circularity)
- **Correlation heatmaps** per class
- **Outlier crop images** — extracted bounding box crops of flagged cells

### Interpretation Caveat
Most "morphological outliers" detected by this system are **image-processing artifacts** rather than genuinely abnormal cells:
- A watershed **under-segmented cluster** appears as one massive cell with absurd area and strange circularity → flagged as outlier.
- A watershed **over-segmented cell fragment** appears as a tiny, irregular shape → flagged as outlier.

This means the outlier detection output is most useful as a **pipeline quality diagnostic** rather than a clinical indicator.

### Outputs
```
results/
    metrics/
        morphology_class_statistics.csv
        morphological_outliers.csv
        outlier_feature_reasons.csv
        outlier_summary.csv
    visualizations/morphology/
        area_by_class.png
        circularity_by_class.png
        shape_relationships/
        color/
        intensity/
        texture/
        outliers/              (crop images of flagged cells)
        outlier_overlays/
        correlation/
    reports/
        morphological_analysis.md
```

---

## 12. Streamlit Web Application

**File:** `app.py`  
**Run:** `streamlit run app.py`

### Features
The web application provides an interactive interface over the complete pipeline:

- **Image Upload** — Drag-and-drop microscopy image upload
- **Automated Analysis** — Runs the frozen pipeline (preprocessing → segmentation → watershed → feature extraction → SVM classification → counting)
- **Results Dashboard** — Displays cell counts (RBC, WBC, Platelets) with a visual summary
- **Cell Inspector** — Browse individual detected cells with their feature values
- **Morphological Charts** — View area, circularity, and color distribution plots
- **Report Download** — Export analysis summary as a text report

### Important Application Disclaimer
The application displays a prominent disclaimer on every analysis:
> *"This is an image-analysis result and is not a clinical blood count or medical diagnosis."*

---

## 13. Evaluation Metrics & Results Summary

### Classification Performance

| Classifier | Macro F1 | Notes |
|---|---|---|
| Rule Baseline | Low | Simple heuristics only |
| KNN (K=5) | High | Good for isolated cells |
| SVM (RBF) | **~1.000** | Best; selected for inference |
| Random Forest | ~1.000 | Comparable to SVM |
| Logistic Regression | Good | Linear; slightly lower |

### Counting Accuracy

| Cell Type | MAE | RMSE | Note |
|---|---|---|---|
| Total | 10.80 | 12.71 | Primary bottleneck: watershed |
| RBC | High error | High | Dense cluster under-segmentation |
| WBC | Low error | Low | Isolated; easier to segment |
| Platelets | Moderate | Moderate | Small size; some missed |

### Object Detection (IoU ≥ 0.5)

| Metric | Value |
|---|---|
| Overall Recall | **~16%** |
| Overall Precision | Moderate |
| Primary Bottleneck | Watershed under-segmentation of dense RBC clusters |

### Watershed Parameter Sensitivity (Validation Set)

| Peak Distance | Threshold | Count MAE | Coverage |
|---|---|---|---|
| 15 | 0.25 | 11.8 | ~99% |
| 40 | 0.35 | **5.9** | ~99.6% |
| 50 | 0.45 | Higher | Lower |

---

## 14. Directory Structure

```
ML Model/
│
├── main.py                    # Main pipeline orchestrator (Stages 1–5, 8)
├── config.py                  # All configuration parameters & path definitions
├── app.py                     # Streamlit web application
├── run_pipeline.py            # CLI runner for pipeline stages
├── requirements.txt
├── README.md
│
├── src/                       # Core source modules
│   ├── __init__.py
│   ├── annotation_utils.py    # YOLO annotation parsing & GT box drawing
│   ├── dataset_utils.py       # File pairing, dataset statistics
│   ├── preprocessing.py       # Color space, blur, CLAHE, illumination
│   ├── segmentation.py        # Otsu, adaptive, K-means, color-range segmentation
│   ├── watershed.py           # Distance transform + watershed separation
│   ├── feature_extraction.py  # 32-feature extraction per cell instance
│   ├── feature_visualization.py
│   ├── classification.py      # Stage 6: ML training & evaluation
│   ├── counting.py            # Stage 7: End-to-end inference & counting
│   ├── morphological_analysis.py # Stage 8: Outlier detection
│   ├── morphology.py          # Morphological operation utilities
│   └── visualization.py       # All figure generation utilities
│
├── notebooks/                 # Jupyter notebooks for exploration
│
└── results/                   # All outputs (auto-created)
    ├── visualizations/
    │   ├── preprocessing/
    │   ├── segmentation/
    │   ├── watershed/
    │   ├── features/
    │   ├── classification/
    │   ├── counting/
    │   └── morphology/
    ├── metrics/               # All CSV metric files
    ├── reports/               # Markdown analysis reports
    └── models/                # Saved ML model files (.joblib)
```

---

## 15. Installation & Reproducibility

### Prerequisites
- Python 3.9+
- TXL-PBC dataset placed at `../TXL-PBC_Dataset-master/TXL-PBC/` relative to the ML Model folder

### Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run complete pipeline on test split
python run_pipeline.py --split test

# Or run individual stages by uncommenting in main.py __main__ block
python main.py
```

### Running the Web App

```bash
streamlit run app.py
```
Access at `http://localhost:8501`

### Stage Execution Order
```
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7 → Stage 8
  ↑         ↑         ↑         ↑          ↑          ↑          ↑         ↑
Dataset  Preproc  Segment  Watershed  Features  Classify  Count   Morphology
Inspect  Analysis  Method              Extract              Stats
```

**Dependency chain:**
- Stage 6 requires Stage 5 output (`cell_features_with_gt.csv`)
- Stage 7 requires Stage 6 output (trained `.joblib` models)
- Stage 8 requires Stage 7 output (`cell_predictions.csv`)

---

## 16. Key Limitations

### 1. Watershed Under-Segmentation (Critical Bottleneck)
Highly dense, overlapping RBC clusters remain merged as single connected components. Watershed can split moderate clusters but fails on severely packed cells. This is the primary driver of:
- Low object detection recall (~16%)
- High total counting MAE

### 2. Bounding Box Ground Truth
TXL-PBC provides bounding boxes, not pixel-level masks. This makes it impossible to compute:
- Pixel-level IoU / Dice / Jaccard scores
- Precise segmentation quality assessment

All coverage metrics are presence-based proxies only.

### 3. Image-Based Metrics ≠ Clinical Values
Cell counts here are relative to the camera's microscope field of view. They are **not** equivalent to:
- Clinical absolute counts (e.g., WBC per µL of blood)
- Differential blood counts used in diagnosis
- Any other standardized hematological measurement

### 4. Staining Variability
The pipeline is tuned for Romanowsky/Giemsa staining (TXL-PBC dataset). Different staining protocols (Wright, May-Grünwald) may alter color distributions and require re-tuning color-based features and thresholds.

### 5. Classical vs. Deep Learning Tradeoff
A YOLO or Mask R-CNN detector would significantly outperform this pipeline in recall, especially for clustered cells. This pipeline prioritizes **interpretability and classical methodology** over raw detection performance.

---

## 17. Technology Stack

| Library | Version | Purpose |
|---|---|---|
| **OpenCV** | ≥ 4.8.0 | Image I/O, color space conversion, filtering, morphology, contour detection |
| **NumPy** | ≥ 1.24.0 | Array operations, distance transforms |
| **SciPy** | ≥ 1.10.0 | Watershed algorithm, hole filling, local maxima detection |
| **scikit-image** | ≥ 0.20.0 | Region properties (regionprops), connected components |
| **scikit-learn** | ≥ 1.2.0 | SVM, Random Forest, KNN, StandardScaler, metrics |
| **Pandas** | ≥ 2.0.0 | Feature DataFrames, CSV export |
| **Matplotlib** | ≥ 3.7.0 | All visualization figures |
| **Streamlit** | ≥ 1.30.0 | Interactive web application |
| **joblib** | ≥ 1.3.0 | Model serialization (.joblib files) |
| **PyYAML** | ≥ 6.0 | Reading data.yaml dataset config |
| **Pillow** | ≥ 9.5.0 | Image I/O utility |

---

## 📜 License & Credits

- **Dataset:** TXL-PBC — Tianjin University Lab Peripheral Blood Cell Dataset
- **Pipeline:** Classical computer vision + ML; educational research use only

---

> **⚠️ Final Reminder:** This system is a **research and educational tool** for automated image analysis. It is **NOT** a medical device, clinical analyzer, or diagnostic system. Results **must not** be used to make any medical or clinical decisions. Always consult qualified medical professionals for blood analysis and health decisions.
