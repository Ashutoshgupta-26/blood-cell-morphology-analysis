import cv2
import numpy as np
from src.preprocessing import load_image, clahe_enhancement, bilateral_filter, grayscale_conversion
from src.morphology import fill_holes
from src.watershed import separate_touching_cells
from src.segmentation import otsu_segmentation
from config import locate_dataset_root

def test_watershed():
    root = locate_dataset_root()
    img = load_image(root / "images/test/7f96e833a77559fb8fe76152b2b76660.png")
    
    green = grayscale_conversion(img, 'green')
    filtered = bilateral_filter(green)
    pre = clahe_enhancement(filtered)
    _, raw = otsu_segmentation(pre)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    filled = fill_holes(closed)
    
    ws = separate_touching_cells(filled, img)
    labels = ws['labels']
    
    print("Num final instances:", labels.max())
    cv2.imwrite("results/debug/rbc_fix/v2/test_ws_41.png", ws['debug_overlay'])
    cv2.imwrite("results/debug/rbc_fix/v2/test_filled_41.png", filled)

if __name__ == "__main__":
    test_watershed()
