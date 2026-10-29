import cv2
import numpy as np
import os
from pathlib import Path
from src.preprocessing import load_image

def method_a_lab(img_bgr):
    """
    Method A: LAB Color Distance
    Calculates distance from a bright background in LAB space.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Assume background is bright, slightly neutral/pinkish
    # Or just use L channel (Lightness) and A channel (Red/Green)
    L = img_lab[:, :, 0]
    A = img_lab[:, :, 1]
    B = img_lab[:, :, 2]
    
    # RBCs are darker (low L) and redder (high A) than plasma
    # Normalize channels
    L_norm = cv2.normalize(255 - L, None, 0, 255, cv2.NORM_MINMAX)  # Invert L so dark is bright
    A_norm = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)
    
    # Combine (dark and red)
    combined = cv2.addWeighted(L_norm, 0.5, A_norm, 0.5, 0)
    
    # Otsu on combined
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return combined, mask

def method_b_hsv(img_bgr):
    """
    Method B: HSV Channels
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    
    # RBCs have higher saturation than background, and lower Value (darker)
    S_norm = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX)
    V_inv = cv2.normalize(255 - V, None, 0, 255, cv2.NORM_MINMAX)
    
    combined = cv2.addWeighted(S_norm, 0.5, V_inv, 0.5, 0)
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return S_norm, mask

def method_c_combined(img_bgr):
    """
    Method C: Color + Intensity Combination
    Combine Green channel (structural contrast) + S channel (color contrast)
    """
    green = img_bgr[:, :, 1]
    green_inv = cv2.normalize(255 - green, None, 0, 255, cv2.NORM_MINMAX)
    
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1]
    S_norm = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX)
    
    combined = cv2.addWeighted(green_inv, 0.5, S_norm, 0.5, 0)
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return combined, mask

if __name__ == "__main__":
    from config import locate_dataset_root
    root = locate_dataset_root()
    img_path = root / "images/test/7f96e833a77559fb8fe76152b2b76660.png"
    img = load_image(img_path)
    
    out_dir = Path("results/debug/rbc_fix/v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "01_original.png"), img)
    
    lab_dist, lab_mask = method_a_lab(img)
    cv2.imwrite(str(out_dir / "lab_color_distance.png"), lab_dist)
    cv2.imwrite(str(out_dir / "02_lab.png"), lab_mask)
    
    hsv_s, hsv_mask = method_b_hsv(img)
    cv2.imwrite(str(out_dir / "hsv_channels.png"), hsv_s)
    cv2.imwrite(str(out_dir / "03_hsv.png"), hsv_mask)
    
    c_dist, c_mask = method_c_combined(img)
    cv2.imwrite(str(out_dir / "04_combined.png"), c_mask)
    
    # Calculate coverage
    boxes = []
    with open(root / "labels/test/7f96e833a77559fb8fe76152b2b76660.txt") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cx, cy, w, h = map(float, parts[1:5])
            h_img, w_img = img.shape[:2]
            x1 = max(0, int((cx - w/2) * w_img))
            y1 = max(0, int((cy - h/2) * h_img))
            x2 = min(w_img, int((cx + w/2) * w_img))
            y2 = min(h_img, int((cy + h/2) * h_img))
            boxes.append((x1, y1, x2, y2))
            
    def eval_cov(m):
        if m is None: return 0, 0
        r = [np.count_nonzero(m[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
        return np.mean(r), sum(1 for x in r if x < 0.2)
        
    print('LAB Coverage:', eval_cov(lab_mask))
    print('HSV Coverage:', eval_cov(hsv_mask))
    print('Combined Coverage:', eval_cov(c_mask))
