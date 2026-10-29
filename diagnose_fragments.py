import cv2
import numpy as np
from pathlib import Path
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter, clahe_enhancement
from src.segmentation import otsu_segmentation, clean_segmentation_mask
from src.watershed import separate_touching_cells
from config import locate_dataset_root

def diagnose_fragments(img_name):
    root = locate_dataset_root()
    img_path = root / "images/test" / img_name
    
    out_dir = Path("results/debug/problem_image")
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "rbc_fragment_analysis"
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    # 01. Original
    img_bgr = load_image(img_path)
    cv2.imwrite(str(out_dir / "01_original.png"), img_bgr)
    
    # 02. Preprocessed
    green = grayscale_conversion(img_bgr, "green")
    filtered = bilateral_filter(green)
    preprocessed = clahe_enhancement(filtered)
    cv2.imwrite(str(out_dir / "02_preprocessed.png"), preprocessed)
    
    # 03. Raw Segmentation
    _, raw_mask = otsu_segmentation(preprocessed)
    cv2.imwrite(str(out_dir / "03_raw_segmentation.png"), raw_mask)
    
    # 04. Cleaned Mask
    cleaned_mask = clean_segmentation_mask(raw_mask)
    cv2.imwrite(str(out_dir / "04_after_morphology.png"), cleaned_mask)
    
    # 05. Connected Components
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    vis_cc = np.zeros_like(img_bgr)
    for i in range(1, num_labels):
        vis_cc[labels_cc == i] = np.random.randint(50, 255, 3)
    cv2.imwrite(str(out_dir / "05_connected_components.png"), vis_cc)
    
    # Watershed
    ws_res = separate_touching_cells(cleaned_mask, img_bgr)
    labels_ws = ws_res["labels"]
    
    # 06. Distance Transform
    vis_dist = cv2.normalize(ws_res["dist_transform"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out_dir / "06_distance_transform.png"), vis_dist)
    
    # 07. Markers
    cv2.imwrite(str(out_dir / "07_markers.png"), ws_res["sure_fg"])
    
    # 08. Watershed raw
    cv2.imwrite(str(out_dir / "08_watershed.png"), np.uint8((labels_ws > 0) * 255))
    
    # 09. Final Instances
    vis_ws = img_bgr.copy()
    for i in range(1, labels_ws.max() + 1):
        mask = np.uint8(labels_ws == i) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_ws, contours, -1, (0, 0, 255), 1)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.putText(vis_ws, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / "09_final_instances.png"), vis_ws)
    print("Saved 9 diagnostic images to results/debug/problem_image/")
    
if __name__ == "__main__":
    diagnose_fragments("7f96e833a77559fb8fe76152b2b76660.png")
