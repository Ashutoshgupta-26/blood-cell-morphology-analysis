import cv2, numpy as np, os, pandas as pd
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter
from src.morphology import fill_holes
from config import locate_dataset_root

def test_triangle():
    root = locate_dataset_root()
    val_images = os.listdir(root / "images/val")[:15]
    
    results = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for img_name in val_images:
        img_path = root / "images/val" / img_name
        lbl_path = root / "labels/val" / img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        if not lbl_path.exists(): continue
        
        img = load_image(img_path)
        g = img[:,:,1]
        
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = max(0, int((cx - w/2) * img.shape[1]))
                    y1 = max(0, int((cy - h/2) * img.shape[0]))
                    x2 = min(img.shape[1], int((cx + w/2) * img.shape[1]))
                    y2 = min(img.shape[0], int((cy + h/2) * img.shape[0]))
                    boxes.append((x1, y1, x2, y2))
                    
        # Triangle
        t_tri, m_tri = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        c_tri = cv2.morphologyEx(m_tri, cv2.MORPH_CLOSE, k)
        f_tri = fill_holes(c_tri)
        num_tri, _ = cv2.connectedComponents(f_tri)
        
        r_tri = [np.count_nonzero(f_tri[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
        cov_tri = np.mean(r_tri) if r_tri else 0
        
        # Otsu
        t_otsu, m_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        c_otsu = cv2.morphologyEx(m_otsu, cv2.MORPH_CLOSE, k)
        f_otsu = fill_holes(c_otsu)
        num_otsu, _ = cv2.connectedComponents(f_otsu)
        
        r_otsu = [np.count_nonzero(f_otsu[y1:y2, x1:x2] == 255) / ((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
        cov_otsu = np.mean(r_otsu) if r_otsu else 0
        
        results.append({
            "Image": img_name,
            "GT_Count": len(boxes),
            "Triangle_Cov": cov_tri,
            "Triangle_Comp": num_tri - 1,
            "Triangle_Thresh": t_tri,
            "Otsu_Cov": cov_otsu,
            "Otsu_Comp": num_otsu - 1,
            "Otsu_Thresh": t_otsu
        })
        
    df = pd.DataFrame(results)
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    test_triangle()
