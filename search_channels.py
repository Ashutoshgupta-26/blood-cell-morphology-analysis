import cv2
import numpy as np
from src.preprocessing import load_image
from src.morphology import fill_holes
from config import locate_dataset_root
import pandas as pd

def search_best_channel_and_close():
    root = locate_dataset_root()
    img = load_image(root / "images/test/7f96e833a77559fb8fe76152b2b76660.png")
    
    h_img, w_img = img.shape[:2]
    
    # Generate channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    channels = {
        "Green": img[:,:,1],
        "HSV_S": hsv[:,:,1],
        "LAB_A": lab[:,:,1],
        "LAB_B_inv": cv2.bitwise_not(lab[:,:,2]),
        "LAB_L": lab[:,:,0]
    }
    
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
                
    results = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    for name, ch in channels.items():
        # Try both normal and inverted Otsu
        for inv in [False, True]:
            mode = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if inv else cv2.THRESH_BINARY + cv2.THRESH_OTSU
            _, mask = cv2.threshold(ch, 0, 255, mode)
            
            # If mask is mostly white, it's the wrong polarity (we want cells to be white, cells are < 50% of image)
            if np.mean(mask) > 127:
                continue
                
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            filled = fill_holes(closed)
            
            covs = [np.count_nonzero(filled[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
            mean_cov = np.mean(covs) if covs else 0
            
            num_comp, _ = cv2.connectedComponents(filled)
            
            results.append({
                "Channel": name,
                "Inverted": inv,
                "Coverage": mean_cov,
                "Components": num_comp - 1
            })
            
            # Save the best one for visualization
            cv2.imwrite(f"results/debug/rbc_fix/v2/test_{name}_{inv}.png", filled)
            
    df = pd.DataFrame(results)
    print(df.sort_values("Coverage", ascending=False))

if __name__ == "__main__":
    search_best_channel_and_close()
