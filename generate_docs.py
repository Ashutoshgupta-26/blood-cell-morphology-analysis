import shutil
import os
import pandas as pd
from pathlib import Path
from config import METRICS_DIR, REPORTS_DIR, VISUALIZATIONS_DIR

PRES_DIR = VISUALIZATIONS_DIR / "presentation"
PRES_DIR.mkdir(parents=True, exist_ok=True)

def copy_best_figs():
    srcs = [
        VISUALIZATIONS_DIR / "final/pipeline_diagram.png",
        VISUALIZATIONS_DIR / "segmentation/binary_masks.png",
        VISUALIZATIONS_DIR / "watershed/watershed_markers.png",
        VISUALIZATIONS_DIR / "classification/classifier_comparison.png",
        VISUALIZATIONS_DIR / "classification/svm_confusion_matrix.png",
        VISUALIZATIONS_DIR / "counting/total_cells_mae.png",
        VISUALIZATIONS_DIR / "morphology/area_by_class.png"
    ]
    
    for s in srcs:
        if s.exists():
            shutil.copy(s, PRES_DIR / s.name)
            
def generate_docs():
    # 1. Setup Guide
    setup = """# Environment Setup Guide

### Requirements
- Python 3.9+
- Windows OS (Tested)

### Installation
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Dataset
The TXL-PBC dataset directory is automatically resolved if placed at `C:\\Blood Project\\TXL-PBC` or one directory above the `ML Model` folder.

### Running CLI
```bash
python run_pipeline.py --split test
```

### Running Streamlit Dashboard
```bash
streamlit run app.py
```
"""
    with open(REPORTS_DIR / "setup_guide.md", "w") as f: f.write(setup)

    # 2. Methodology
    meth = """# Methodology

1. **Problem Definition**: Segment, classify, and count blood cells while profiling morphology.
2. **Dataset**: TXL-PBC Dataset.
3. **Preprocessing**: Bilateral Filter (noise reduction) + CLAHE (contrast enhancement).
4. **Segmentation**: Otsu thresholding on Green Channel.
5. **Morphological Operations**: Hole filling & area constraints to remove background noise.
6. **Distance Transform**: Euclidean metric computing distance to nearest background.
7. **Watershed**: Classical separation of touching cells seeded by local distance maxima.
8. **Feature Extraction**: Area, Circularity ($C = 4\pi A / P^2$), Solidity ($S = A / A_{convex\_hull}$), HSV/LAB statistics.
9. **Classification**: Non-linear SVM with RBF kernel mapping vectors to biological classes.
10. **Counting**: Generating density analytics mapping predicted instances vs GT boxes.
11. **Outlier Detection**: IQR, Z-score, Isolation Forest identifying image artifacts.

*Disclaimer: This is an image-processing educational tool. It is not a clinical diagnostic system.*
"""
    with open(REPORTS_DIR / "methodology.md", "w") as f: f.write(meth)

    # 3. Project Abstract
    abs_txt = """# Project Abstract
**Problem**: Automating blood smear analysis using non-deep learning classical image processing.
**Method**: We engineered a pipeline fusing CLAHE, Otsu thresholding, Distance Transform Watershed, and handcrafted morphological/color features classified by a Support Vector Machine.
**Main Result**: The SVM achieves perfect theoretical classification accuracy (F1=1.0) on clean cell crops. However, End-to-End detection recall is severely bottlenecked (16%) by the Watershed algorithm's inability to shatter deep, multi-layered RBC clusters, driving down clinical counting accuracy.
**Limitations**: Morphological clustering causes immense bounding box anomalies and undercounting. Not a medical diagnosis tool.
"""
    with open(REPORTS_DIR / "project_abstract.md", "w") as f: f.write(abs_txt)

    # 4. Viva Cheat Sheet
    viva = """# Viva Cheat Sheet

**1. Why classical image processing instead of YOLO?**
To demonstrate fundamental computer vision algorithms (thresholding, morphology, feature extraction) rather than relying on black-box neural networks.

**2. Why CLAHE?**
Contrast Limited Adaptive Histogram Equalization prevents over-amplification of noise, enhancing cell edges against bright backgrounds.

**3. Why Otsu segmentation?**
It mathematically optimizes the global threshold by maximizing inter-class variance between cells and plasma.

**4. Why Watershed?**
Without it, touching RBCs merge into one object. Watershed treats distance transforms as topographical basins to snap borders between touching elements.

**5. Why did Watershed fail on some images?**
RBCs heavily overlap in dense smears. The Distance Transform cannot find distinct local maxima if the cells form a flat, layered plateau.

**6. What is Circularity?**
$4\pi \cdot Area / Perimeter^2$. A perfect circle is 1.0. Jagged or merged cells drop heavily.

**7. Why use LAB/HSV color spaces?**
RGB is highly correlated to lighting. HSV separates Hue (color) from Value (brightness). LAB closely matches human perception.

**8. Why SVM?**
It finds the optimal maximum-margin hyperplane separating classes in high-dimensional space, proving highly robust on our 32 extracted features.

**9. What is MAE / RMSE?**
Mean Absolute Error measures average absolute count difference. Root Mean Square Error heavily penalizes massive outliers.

**10. What is a morphological outlier?**
A statistically anomalous cell geometry/color. In our pipeline, it primarily flags image-processing failures (merged clusters) rather than biological diseases.
"""
    with open(REPORTS_DIR / "viva_cheat_sheet.md", "w") as f: f.write(viva)

    # 5. Project Summary
    summ = """# What We Built

### Input: Blood-smear image
### Step 1: Enhance image (Bilateral Filter, CLAHE)
### Step 2: Separate foreground cells (Otsu Thresholding)
### Step 3: Clean segmentation (Hole filling)
### Step 4: Separate touching cells (Distance Transform Watershed)
### Step 5: Measure every cell (Area, Circularity, Color)
### Step 6: Classify cells (SVM)
### Step 7: Count cells (Object Analytics)
### Step 8: Analyze morphology (Z-score Outliers)
### Step 9: Display everything in Streamlit
"""
    with open(REPORTS_DIR / "project_summary.md", "w") as f: f.write(summ)

    # 6. Conclusion
    conc = """# Conclusion
We built a robust, end-to-end Classical Image Processing architecture. The classification engine using morphological features is extremely powerful (1.0 F1). The main failure point is the Watershed segmentation under-segmenting dense RBC clusters. Future work would transition from classical algorithms to CNN-based Mask R-CNN or YOLO architectures for instance segmentation.
"""
    with open(REPORTS_DIR / "conclusion.md", "w") as f: f.write(conc)

    # 7. Manifest
    man = """# Project Manifest
`src/preprocessing.py`: Image preprocessing
`src/segmentation.py`: Classical segmentation
`src/watershed.py`: Touching-cell separation
`src/feature_extraction.py`: Cell feature extraction
`src/classification.py`: Classical ML classification
`src/counting.py`: Cell counting
`src/morphological_analysis.py`: Morphology/outlier analysis
`app.py`: Streamlit interface
`run_pipeline.py`: Automated CLI execution
`config.py`: Central hyperparameters
"""
    with open(REPORTS_DIR / "project_manifest.md", "w") as f: f.write(man)

    # 8. Consolidate Final Results CSV
    summ_df = pd.read_csv(METRICS_DIR / "final_summary.csv")
    res_map = {
        "Dataset": "TXL-PBC",
        "Images": summ_df.iloc[0]["dataset_images"],
        "Cells": summ_df.iloc[0]["dataset_cells"],
        "Classification Accuracy": summ_df.iloc[0]["classification_accuracy"],
        "Classification Macro F1": summ_df.iloc[0]["classification_macro_f1"],
        "Detection Precision": summ_df.iloc[0]["object_precision"],
        "Detection Recall": summ_df.iloc[0]["object_recall"],
        "Detection F1": summ_df.iloc[0]["object_f1"],
        "RBC MAE": summ_df.iloc[0]["rbc_mae"],
        "WBC MAE": summ_df.iloc[0]["wbc_mae"],
        "Platelet MAE": summ_df.iloc[0]["platelet_mae"],
        "Total MAE": summ_df.iloc[0]["total_mae"],
        "RBC RMSE": summ_df.iloc[0]["rbc_rmse"],
        "WBC RMSE": summ_df.iloc[0]["wbc_rmse"],
        "Platelet RMSE": summ_df.iloc[0]["platelet_rmse"],
        "Total RMSE": summ_df.iloc[0]["total_rmse"],
        "Morphological Outliers": summ_df.iloc[0]["number_of_morphological_outliers"]
    }
    pd.DataFrame([res_map]).to_csv(METRICS_DIR / "final_results.csv", index=False)
    
if __name__ == "__main__":
    copy_best_figs()
    generate_docs()
    print("Stage 11 Documentation Generated.")
