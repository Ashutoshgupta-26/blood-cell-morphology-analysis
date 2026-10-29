"""
Fine-tune: 
1. Filter edge-touching instances (border objects)
2. Add quality filtering for low-circularity / low-solidity detections
3. Check if missed GTs are border cells or segmentation failures
"""
import cv2, numpy as np
from src.preprocessing import load_image
from src.counting import load_classification_pipeline, analyze_blood_smear
from src.segmentation import segment_cells
from config import locate_dataset_root

root = locate_dataset_root()
img = load_image(root / "images/test/7f96e833a77559fb8fe76152b2b76660.png")
h_img, w_img = img.shape[:2]

s, m, fc = load_classification_pipeline()
r = analyze_blood_smear(str(root / "images/test/7f96e833a77559fb8fe76152b2b76660.png"), s, m, fc)
df = r['predictions_df']
labels = r['labels']

print("=" * 70)
print("EDGE ANALYSIS (FP candidates)")
print("=" * 70)
MARGIN = 5
for _, row in df.iterrows():
    lid = int(row['instance_id'])
    mask = (labels == lid).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: continue
    
    touches_top = ys.min() <= MARGIN
    touches_bot = ys.max() >= h_img - MARGIN - 1
    touches_left = xs.min() <= MARGIN
    touches_right = xs.max() >= w_img - MARGIN - 1
    touches_edge = touches_top or touches_bot or touches_left or touches_right
    
    edge_str = ""
    if touches_top: edge_str += "T"
    if touches_bot: edge_str += "B"
    if touches_left: edge_str += "L"
    if touches_right: edge_str += "R"
    
    if touches_edge or row['area'] < 500 or row['circularity'] < 0.35 or row['solidity'] < 0.7:
        flag = "EDGE" if touches_edge else ""
        flag += " SMALL" if row['area'] < 500 else ""
        flag += " LOW_CIRC" if row['circularity'] < 0.35 else ""
        flag += " LOW_SOL" if row['solidity'] < 0.7 else ""
        print(f"  #{lid:2d} {row['predicted_class']:>8} area={row['area']:6.0f} circ={row['circularity']:.2f} sol={row['solidity']:.2f} edge={edge_str:4s} => {flag}")

print("\n" + "=" * 70)
print("MISSED GT BOXES ANALYSIS")
print("=" * 70)
# GT #10: RBC @ (130,367) box=[78,321,182,414]
# GT #20: RBC @ (396,354) box=[354,312,438,396]
seg = segment_cells(img)
cleaned = seg['cleaned_mask']

for gt_id, (x1,y1,x2,y2) in [(10, (78,321,182,414)), (20, (354,312,438,396))]:
    region = cleaned[y1:y2, x1:x2]
    cov = np.count_nonzero(region == 255) / ((x2-x1)*(y2-y1))
    print(f"  GT #{gt_id}: coverage={cov:.3f} in mask")
