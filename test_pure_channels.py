import cv2
import numpy as np
from src.preprocessing import load_image
from config import locate_dataset_root
from pathlib import Path

def evaluate_pure_channels():
    root = locate_dataset_root()
    img_path = root / "images/test/7f96e833a77559fb8fe76152b2b76660.png"
    img = load_image(img_path)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    s = hsv[:, :, 1]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # Invert B because background (126) > cells (79)
    b_inv = cv2.bitwise_not(b)
    
    # Otsu thresholds
    _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_b = cv2.threshold(b_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
        r = [np.count_nonzero(m[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
        return np.mean(r), sum(1 for x in r if x < 0.2), sum(1 for x in r if x > 0.6)
        
    print('S Channel:', eval_cov(mask_s))
    print('A Channel:', eval_cov(mask_a))
    print('B Channel (Inv):', eval_cov(mask_b))

if __name__ == "__main__":
    evaluate_pure_channels()
