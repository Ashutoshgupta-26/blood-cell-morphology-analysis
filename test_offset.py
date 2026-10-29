import cv2
import numpy as np
from src.preprocessing import load_image, clahe_enhancement, bilateral_filter, grayscale_conversion
from src.morphology import fill_holes
from config import locate_dataset_root

def test_otsu_offset():
    root = locate_dataset_root()
    img = load_image(root / "images/test/7f96e833a77559fb8fe76152b2b76660.png")
    h_img, w_img = img.shape[:2]
    
    pre = clahe_enhancement(bilateral_filter(grayscale_conversion(img, 'green')))
    
    # Standard Otsu
    thresh_val, _ = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Try different offsets
    offsets = [1.0, 0.9, 0.8, 0.7]
    for off in offsets:
        # Note: the background is bright in clahe_green. So inverted means foreground is > thresh? No, foreground is < thresh.
        # Wait, if we use THRESH_BINARY_INV, pixels < thresh become 255.
        # So to get MORE foreground, we INCREASE the threshold!
        # Let's see what normalize_foreground_polarity does.
        
        # Let's just use Otsu without INV, and let polarity normalizer handle it.
        t, raw = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adjust threshold
        adjusted_t = t * off
        _, mask = cv2.threshold(pre, adjusted_t, 255, cv2.THRESH_BINARY)
        
        # Invert if needed (assuming cells are dark)
        # Actually clahe_green makes cells dark. So THRESH_BINARY_INV is correct.
        _, mask_inv = cv2.threshold(pre, adjusted_t, 255, cv2.THRESH_BINARY_INV)
        
        filled = fill_holes(mask_inv)
        
        # Calculate coverage
        boxes = []
        with open(root / "labels/test/7f96e833a77559fb8fe76152b2b76660.txt") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = max(0, int((cx - w/2) * w_img))
                    y1 = max(0, int((cy - h/2) * h_img))
                    x2 = min(w_img, int((cx + w/2) * w_img))
                    y2 = min(h_img, int((cy + h/2) * h_img))
                    boxes.append((x1, y1, x2, y2))
                    
        ratios = [np.count_nonzero(filled[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
        
        print(f"Offset {off} (Thresh {adjusted_t:.1f}): Mean Cov = {np.mean(ratios):.4f}, Bad = {sum(1 for r in ratios if r < 0.2)}")

if __name__ == "__main__":
    test_otsu_offset()
