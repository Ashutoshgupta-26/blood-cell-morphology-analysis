"""
Configuration settings for Blood Cell Analysis Project.
Provides dataset paths, class mappings, visualization colors, output directory definitions,
and algorithm parameters for preprocessing and segmentation stages.
"""

import sys
from pathlib import Path
import yaml

# Base directory of the ML Model project
BASE_DIR = Path(__file__).resolve().parent

# Results directory structure
RESULTS_DIR = BASE_DIR / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
PREPROCESSING_VIS_DIR = VISUALIZATIONS_DIR / "preprocessing"
SEGMENTATION_VIS_DIR = VISUALIZATIONS_DIR / "segmentation"
WATERSHED_VIS_DIR = VISUALIZATIONS_DIR / "watershed"
MASKS_DIR = RESULTS_DIR / "masks"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"


def ensure_directories():
    """Ensure all required output directories exist."""
    for path in [
        RESULTS_DIR,
        VISUALIZATIONS_DIR,
        PREPROCESSING_VIS_DIR,
        SEGMENTATION_VIS_DIR,
        WATERSHED_VIS_DIR,
        MASKS_DIR,
        METRICS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def locate_dataset_root(custom_path=None):
    """
    Automatically search for and locate the TXL-PBC dataset root directory.
    
    Returns:
        Path: Path object pointing to the dataset root containing data.yaml and images/
    """
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    candidates = [
        BASE_DIR.parent / "TXL-PBC_Dataset-master" / "TXL-PBC",
        BASE_DIR.parent / "TXL-PBC",
        Path("c:/Blood Project/TXL-PBC_Dataset-master/TXL-PBC"),
        Path("c:/Blood Project/TXL-PBC"),
        BASE_DIR / "data" / "TXL-PBC",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "data.yaml").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not automatically locate TXL-PBC dataset root with data.yaml. "
        "Please specify dataset path in config.py or pass custom_path."
    )


# Attempt dataset resolution
try:
    DATASET_ROOT = locate_dataset_root()
except FileNotFoundError:
    DATASET_ROOT = None

# Default Class Definitions matching TXL-PBC dataset
# ID 0: WBC (White Blood Cell), ID 1: RBC (Red Blood Cell), ID 2: Platelets
CLASS_ID_TO_NAME = {
    0: "WBC",
    1: "RBC",
    2: "Platelets",
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}

# Colors for bounding box visualizations (BGR format for OpenCV, RGB for Matplotlib)
CLASS_COLORS_BGR = {
    0: (255, 50, 150),   # WBC (Purple/Magenta)
    1: (50, 50, 220),    # RBC (Red)
    2: (0, 200, 255),    # Platelets (Yellow/Orange)
}

CLASS_COLORS_RGB = {
    0: (150, 50, 255),   # WBC
    1: (220, 50, 50),    # RBC
    2: (255, 200, 0),    # Platelets
}

# Stage 2: Image Preprocessing Configuration Parameters
PREPROCESSING_CONFIG = {
    "gaussian_kernel": (5, 5),
    "gaussian_sigma": 1.0,
    "median_kernel": 5,
    "bilateral_d": 9,
    "bilateral_sigma_color": 75.0,
    "bilateral_sigma_space": 75.0,
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),
    "background_blur_kernel": (51, 51),
    "sample_size": 10,
}

# Stage 3: Cell Segmentation Configuration Parameters
SEGMENTATION_CONFIG = {
    # Preferred Preprocessing Input Channel ('green', 'clahe_green', 'grayscale', 'hsv_s', 'lab_l')
    "input_channel": "green",
    # Default Segmentation Algorithm ('otsu', 'adaptive', 'color_range', 'kmeans', 'triangle')
    "segmentation_method": "triangle",
    # Adaptive thresholding settings
    "adaptive_method": "gaussian",
    "adaptive_block_size": 21,
    "adaptive_c": 5.0,
    # HSV color thresholding range (Saturation channel)
    "hsv_s_lower": 30,
    "hsv_s_upper": 255,
    # K-Means clustering clusters
    "kmeans_k": 3,
    # Morphological cleanup settings
    "morph_kernel_size": (3, 3),
    "min_component_area": 100,       # Preserves small cells (platelets) while eliminating tiny noise (Min Platelet Area = 133)
    "remove_border_objects": False,  # Default False: avoid deleting real cells touching the edge
}

# Stage 4: Watershed Segmentation Configuration Parameters
WATERSHED_CONFIG = {
    # Peak spacing is scaled from this reference short-edge image size.
    "min_peak_distance": 40,
    "reference_short_edge": 480,
    # Validation sweep: 0.35 slightly reduced count error and spurious
    # instance splits compared with 0.25 on the current held-out setup.
    "distance_threshold_ratio": 0.35,
    "min_cell_area": 20, # Retain small platelet-sized regions after watershed
}

def load_dataset_classes(data_yaml_path=None):
    """
    Load exact class definitions from data.yaml if available.
    """
    if data_yaml_path is None and DATASET_ROOT is not None:
        data_yaml_path = DATASET_ROOT / "data.yaml"

    if data_yaml_path and Path(data_yaml_path).exists():
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)
            names = data_yaml.get("names", [])
            class_map = {idx: name.strip() for idx, name in enumerate(names)}
            return class_map

    return CLASS_ID_TO_NAME
