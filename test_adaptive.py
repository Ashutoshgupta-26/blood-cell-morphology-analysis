import cv2
import numpy as np
from src.preprocessing import load_image
from config import locate_dataset_root

root = locate_dataset_root()
img = load_image(root / "images/test/7f96e833a77559fb8fe76152b2b76660.png")
g = img[:,:,1]

# Adaptive
th_adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)

boxes = []
with open(root / "labels/test/7f96e833a77559fb8fe76152b2b76660.txt") as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            cx, cy, w, h = map(float, parts[1:5])
            h_img, w_img = img.shape[:2]
            x1 = max(0, int((cx - w/2) * w_img))
            y1 = max(0, int((cy - h/2) * h_img))
            x2 = min(w_img, int((cx + w/2) * w_img))
            y2 = min(h_img, int((cy + h/2) * h_img))
            boxes.append((x1, y1, x2, y2))

def eval_cov(m):
    r = [np.count_nonzero(m[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
    return np.mean(r)

print('Adaptive:', eval_cov(th_adapt))
print('Adaptive Mean:', np.mean(th_adapt))
