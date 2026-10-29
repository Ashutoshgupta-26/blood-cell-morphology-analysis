"""Test different min_peak_distance values for watershed."""
import cv2, numpy as np, os
from src.preprocessing import load_image, grayscale_conversion, bilateral_filter
from src.segmentation import segment_cells
from src.watershed import separate_touching_cells
from config import locate_dataset_root

root = locate_dataset_root()
test_images = [
    "7f96e833a77559fb8fe76152b2b76660.png",
]
# Also grab 5 val images
val_images = sorted(os.listdir(root / "images/val"))[:5]

for min_d in [10, 12, 15, 18, 20]:
    total_comp = 0
    for img_name in test_images:
        img = load_image(root / "images/test" / img_name)
        seg = segment_cells(img)
        ws = separate_touching_cells(seg['cleaned_mask'], img, threshold_ratio=None)
        # Override min_distance manually
        from src.watershed import compute_distance_transform, generate_foreground_markers, generate_background_markers
        dt = compute_distance_transform(seg['cleaned_mask'])
        sf, markers, nl = generate_foreground_markers(dt, seg['cleaned_mask'], min_distance=min_d)
        sb = generate_background_markers(seg['cleaned_mask'])
        unknown = cv2.subtract(sb, sf)
        mc = markers.copy() + 1
        mc[unknown == 255] = 0
        mw = cv2.watershed(img.copy(), mc)
        unique_labels = set(np.unique(mw)) - {-1, 0, 1}
        total_comp += len(unique_labels)
    
    for img_name in val_images:
        img = load_image(root / "images/val" / img_name)
        seg = segment_cells(img)
        dt = compute_distance_transform(seg['cleaned_mask'])
        sf, markers, nl = generate_foreground_markers(dt, seg['cleaned_mask'], min_distance=min_d)
        sb = generate_background_markers(seg['cleaned_mask'])
        unknown = cv2.subtract(sb, sf)
        mc = markers.copy() + 1
        mc[unknown == 255] = 0
        mw = cv2.watershed(img.copy(), mc)
        unique_labels = set(np.unique(mw)) - {-1, 0, 1}
        total_comp += len(unique_labels)
    
    avg = total_comp / (len(test_images) + len(val_images))
    print(f"min_distance={min_d:2d}: avg instances={avg:.1f}")
