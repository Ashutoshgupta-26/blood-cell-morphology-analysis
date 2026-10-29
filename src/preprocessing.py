"""
Image Preprocessing Pipeline for Blood Smear Microscopy Images (Stage 2).
Provides modular functions for color space conversion, noise filtering, contrast enhancement,
illumination correction, image statistics calculation, and complete pipeline execution.
"""

import sys
from pathlib import Path
import math
import cv2
import numpy as np

# Add parent directory to path to load config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PREPROCESSING_CONFIG


def load_image(image_path):
    """
    Load a blood smear image from file path using OpenCV.
    
    Args:
        image_path (str or Path): Path to image file.
        
    Returns:
        np.ndarray: BGR image array.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image file: {image_path}")

    return img_bgr


def convert_color_spaces(image_bgr):
    """
    Convert a BGR image into multiple color spaces and extract individual channels.
    
    Color Spaces Explored:
    - BGR / RGB: Standard color spaces
    - HSV: Hue (color tint), Saturation (color purity/depth), Value (brightness)
    - LAB: Lightness (L), green-red component (a), blue-yellow component (b)
    - Grayscale: Luminance representation
    
    Args:
        image_bgr (np.ndarray): BGR image array.
        
    Returns:
        dict: Mapping of color space names and individual channel arrays.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(image_bgr)
    h, s, v = cv2.split(img_hsv)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    return {
        "BGR": image_bgr,
        "RGB": img_rgb,
        "HSV": img_hsv,
        "LAB": img_lab,
        "GRAY": img_gray,
        # Individual Channels
        "R_channel": r,
        "G_channel": g,
        "B_channel": b,
        "H_channel": h,
        "S_channel": s,
        "V_channel": v,
        "L_channel": l_chan,
        "A_channel": a_chan,
        "B_lab_channel": b_chan,
    }


def grayscale_conversion(image_bgr, method="green"):
    """
    Convert a BGR image to single-channel grayscale using different strategies.
    
    Strategies:
    - 'standard': Weighted luminance conversion 0.299 R + 0.587 G + 0.114 B.
    - 'green': Green channel extraction. Green light is strongly absorbed by hemoglobin 
               and nuclear Giemsa/Romanowsky stain, providing optimal cell/background contrast.
    - 'lab_l': Lightness channel (L) from LAB color space.
    - 'hsv_s': Saturation channel (S) from HSV color space.
    
    Args:
        image_bgr (np.ndarray): BGR image array.
        method (str): Strategy name ('standard', 'green', 'lab_l', 'hsv_s').
        
    Returns:
        np.ndarray: 2D uint8 single-channel image.
    """
    if method == "standard":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif method == "green":
        # Green channel (index 1 in BGR format)
        return image_bgr[:, :, 1]
    elif method == "lab_l":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]
    elif method == "lab_a":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 1]
    elif method == "hsv_s":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]
    else:
        raise ValueError(f"Unknown grayscale conversion method: {method}")


def gaussian_blur(image, kernel_size=None, sigma=None):
    """
    Apply Gaussian Smoothing Blur to reduce high-frequency noise.
    
    Args:
        image (np.ndarray): 2D or 3D image array.
        kernel_size (tuple, optional): Kernel dimensions (width, height). Defaults to config.
        sigma (float, optional): Gaussian kernel standard deviation. Defaults to config.
        
    Returns:
        np.ndarray: Smoothed image array.
    """
    if kernel_size is None:
        kernel_size = PREPROCESSING_CONFIG.get("gaussian_kernel", (5, 5))
    if sigma is None:
        sigma = PREPROCESSING_CONFIG.get("gaussian_sigma", 1.0)

    # Ensure kernel dimensions are odd integers
    k_w = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_h = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1

    return cv2.GaussianBlur(image, (k_w, k_h), sigmaX=sigma, sigmaY=sigma)


def median_blur(image, kernel_size=None):
    """
    Apply Non-Linear Median Filter to suppress impulse (salt-and-pepper) noise while preserving sharp boundaries.
    
    Args:
        image (np.ndarray): 2D uint8 image array.
        kernel_size (int, optional): Square kernel size. Defaults to config.
        
    Returns:
        np.ndarray: Filtered image array.
    """
    if kernel_size is None:
        kernel_size = PREPROCESSING_CONFIG.get("median_kernel", 5)

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(image, d=None, sigma_color=None, sigma_space=None):
    """
    Apply Bilateral Filter for edge-preserving smoothing.
    Smoothes flat regions (like plasma background) while maintaining sharp intensity transitions at cell borders.
    
    Args:
        image (np.ndarray): 2D or 3D uint8 image array.
        d (int, optional): Diameter of pixel neighborhood.
        sigma_color (float, optional): Filter sigma in color space.
        sigma_space (float, optional): Filter sigma in coordinate space.
        
    Returns:
        np.ndarray: Filtered image array.
    """
    if d is None:
        d = PREPROCESSING_CONFIG.get("bilateral_d", 9)
    if sigma_color is None:
        sigma_color = PREPROCESSING_CONFIG.get("bilateral_sigma_color", 75.0)
    if sigma_space is None:
        sigma_space = PREPROCESSING_CONFIG.get("bilateral_sigma_space", 75.0)

    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def histogram_equalization(image_gray):
    """
    Apply Global Histogram Equalization to stretch gray-level dynamic range across the full [0, 255] spectrum.
    
    Note: While effective for globally dark/bright images, global histogram equalization can over-amplify noise
    or distort contrast in medical images with non-uniform illumination.
    
    Args:
        image_gray (np.ndarray): 2D uint8 grayscale image.
        
    Returns:
        np.ndarray: Equalized grayscale image.
    """
    if len(image_gray.shape) != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image_gray)


def clahe_enhancement(image_gray, clip_limit=None, tile_grid_size=None):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Why CLAHE is preferred over Global Histogram Equalization for blood smear images:
    1. Local Adaptive Enhancement: Operates on small contextual regions (tiles), compensating for non-uniform illumination across the microscope field of view.
    2. Contrast Limiting: Clips histogram peaks to avoid over-amplifying background noise or burning out delicate cell interior textures.
    
    Args:
        image_gray (np.ndarray): 2D uint8 grayscale image.
        clip_limit (float, optional): Contrast limit threshold. Defaults to config.
        tile_grid_size (tuple, optional): Grid size for contextual tiles (e.g. 8x8). Defaults to config.
        
    Returns:
        np.ndarray: Enhanced grayscale image.
    """
    if len(image_gray.shape) != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    if clip_limit is None:
        clip_limit = PREPROCESSING_CONFIG.get("clahe_clip_limit", 2.0)
    if tile_grid_size is None:
        tile_grid_size = PREPROCESSING_CONFIG.get("clahe_grid_size", (8, 8))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_gray)


def correct_illumination(image_gray, kernel_size=None):
    """
    Estimate and normalize non-uniform background illumination across the image field.
    
    Method:
    1. Estimate background illumination using a large Gaussian blur kernel.
    2. Subtract estimated background or divide to normalize lighting unevenness.
    
    Args:
        image_gray (np.ndarray): 2D uint8 grayscale image.
        kernel_size (tuple, optional): Large Gaussian kernel dimensions. Defaults to (51, 51).
        
    Returns:
        np.ndarray: Illumination-corrected uint8 image array.
    """
    if len(image_gray.shape) != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    if kernel_size is None:
        kernel_size = PREPROCESSING_CONFIG.get("background_blur_kernel", (51, 51))

    # Large blur to capture low-frequency background illumination map
    bg_estimated = cv2.GaussianBlur(image_gray, kernel_size, sigmaX=0, sigmaY=0)

    # Normalize image by dividing image by background estimate (scaled to 255)
    img_float = image_gray.astype(np.float32) + 1.0
    bg_float = bg_estimated.astype(np.float32) + 1.0

    corrected_float = (img_float / bg_float) * 128.0
    corrected = np.clip(corrected_float, 0, 255).astype(np.uint8)

    return corrected


def calculate_shannon_entropy(image):
    """
    Calculate Shannon Entropy of an image (measure of information/randomness).
    
    Args:
        image (np.ndarray): 2D uint8 image array.
        
    Returns:
        float: Shannon entropy in bits.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate 256-bin histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()

    # Compute entropy: -sum(p * log2(p)) for p > 0
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    return float(entropy)


def compute_image_statistics(image, method_name="Raw", image_name=""):
    """
    Calculate comprehensive quantitative statistics for an image array.
    
    Args:
        image (np.ndarray): Image array (grayscale or BGR).
        method_name (str): Name of preprocessing method applied.
        image_name (str): Identifier/filename of image.
        
    Returns:
        dict: Statistical metrics including mean, std, min, max, contrast (RMS), and entropy.
    """
    if len(image.shape) == 3:
        # For 3-channel images, compute overall gray conversion for unified stats
        gray_equivalent = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_equivalent = image

    mean_val = float(np.mean(gray_equivalent))
    std_val = float(np.std(gray_equivalent))
    min_val = int(np.min(gray_equivalent))
    max_val = int(np.max(gray_equivalent))
    contrast_rms = std_val  # RMS Contrast is defined as standard deviation of pixel intensities
    entropy_val = calculate_shannon_entropy(gray_equivalent)

    return {
        "image_name": image_name,
        "method": method_name,
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": min_val,
        "max": max_val,
        "contrast_rms": round(contrast_rms, 2),
        "entropy": round(entropy_val, 3),
    }


def preprocess_image(image_bgr, pipeline_options=None):
    """
    Execute complete recommended preprocessing pipeline on a blood smear image.
    
    Recommended Pipeline Strategy:
    1. Extract Green Channel (optimal blood cell vs plasma contrast).
    2. Apply Bilateral Filter (reduces plasma noise while preserving sharp cell boundaries).
    3. Apply CLAHE (enhances local cell contrast without over-amplifying noise).
    
    Args:
        image_bgr (np.ndarray): Input BGR image array.
        pipeline_options (dict, optional): Custom override options.
        
    Returns:
        dict: Preprocessing results including intermediate steps and final enhanced image.
    """
    if pipeline_options is None:
        pipeline_options = {}

    # Step 1: Grayscale / Channel Selection (Green channel recommended)
    gray_method = pipeline_options.get("grayscale_method", "green")
    gray_step = grayscale_conversion(image_bgr, method=gray_method)

    # Step 2: Noise Reduction (Bilateral Filter recommended)
    filter_type = pipeline_options.get("filter_type", "bilateral")
    if filter_type == "bilateral":
        filtered_step = bilateral_filter(gray_step)
    elif filter_type == "gaussian":
        filtered_step = gaussian_blur(gray_step)
    elif filter_type == "median":
        filtered_step = median_blur(gray_step)
    else:
        filtered_step = gray_step

    # Step 3: Contrast Enhancement (CLAHE recommended)
    contrast_type = pipeline_options.get("contrast_type", "clahe")
    if contrast_type == "clahe":
        enhanced_step = clahe_enhancement(filtered_step)
    elif contrast_type == "equalization":
        enhanced_step = histogram_equalization(filtered_step)
    else:
        enhanced_step = filtered_step

    return {
        "original_bgr": image_bgr,
        "gray_step": gray_step,
        "filtered_step": filtered_step,
        "final_enhanced": enhanced_step,
    }
