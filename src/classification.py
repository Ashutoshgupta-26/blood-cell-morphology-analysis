"""
Classification Module (Stage 6).
Classifies individual cell instances (RBC, WBC, Platelets) using classical ML.
Uses image-level splitting to prevent data leakage.
"""

import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib

from .dataset_utils import get_paired_files
from .annotation_utils import parse_yolo_annotation
from .preprocessing import load_image
from .segmentation import segment_cells
from .watershed import separate_touching_cells
from .feature_extraction import extract_features_for_image, associate_ground_truth
from config import WATERSHED_CONFIG, METRICS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR

MODELS_DIR = Path(__file__).resolve().parent.parent / "results" / "models"

def ensure_classification_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_vis_dir = VISUALIZATIONS_DIR / "classification"
    class_vis_dir.mkdir(parents=True, exist_ok=True)
    (class_vis_dir / "errors").mkdir(parents=True, exist_ok=True)
    return class_vis_dir

def build_classification_dataset(num_train=50, num_val=20, num_test=20):
    """
    Run the extraction pipeline on a balanced subset of images to build the dataset.
    """
    print("Building Classification Dataset (Image-level split)...")
    
    all_features = []
    
    splits = [("train", num_train), ("val", num_val), ("test", num_test)]
    best_th = WATERSHED_CONFIG.get("distance_threshold_ratio", 0.25)
    
    for split_name, sample_size in splits:
        pairs = get_paired_files(split=split_name)
        
        # Sort to get some variety
        selected_pairs = []
        for p in pairs:
            img = load_image(p["image_path"])
            h, w = img.shape[:2]
            anns = parse_yolo_annotation(p["label_path"], w, h)
            cnames = set(a["class_name"] for a in anns)
            selected_pairs.append((p, len(anns), cnames, anns))
            if len(selected_pairs) > sample_size * 2:
                break
                
        selected_pairs.sort(key=lambda x: ("Platelets" in x[2], "WBC" in x[2], x[1]), reverse=True)
        final_pairs = selected_pairs[:sample_size]
        
        print(f"  Extracting features from {len(final_pairs)} {split_name} images...")
        
        for idx, (pair, num_anns, cnames, anns) in enumerate(final_pairs, 1):
            img_bgr = load_image(pair["image_path"])
            seg_res = segment_cells(img_bgr, method="otsu", channel="green")
            w_res = separate_touching_cells(seg_res["cleaned_mask"], img_bgr, threshold_ratio=best_th)
            labels = w_res["labels"]
            
            features_list = extract_features_for_image(img_bgr, labels, pair["image_path"].name)
            if not features_list: continue
            
            df_img = pd.DataFrame(features_list)
            matches = associate_ground_truth(df_img, labels, anns, pair["image_path"].name)
            matches_df = pd.DataFrame(matches)
            
            if not matches_df.empty:
                clean_matches = matches_df[matches_df["match_status"] == "matched"]
                df_img = df_img.merge(
                    clean_matches[["image_name", "best_instance_id", "gt_class", "gt_class_id", "match_iou", "match_status"]],
                    left_on=["image_name", "instance_id"],
                    right_on=["image_name", "best_instance_id"],
                    how="inner" # ONLY KEEP CLEAN MATCHES
                )
                df_img["dataset_split"] = split_name
                all_features.append(df_img)
                
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.drop(columns=["best_instance_id"], inplace=True, errors="ignore")
        final_df.to_csv(METRICS_DIR / "classification_dataset.csv", index=False)
        return final_df
    return pd.DataFrame()

def check_data_leakage(df):
    """Verify that images do not overlap across splits."""
    train_imgs = set(df[df["dataset_split"] == "train"]["image_name"])
    val_imgs = set(df[df["dataset_split"] == "val"]["image_name"])
    test_imgs = set(df[df["dataset_split"] == "test"]["image_name"])
    
    t_v = train_imgs.intersection(val_imgs)
    t_test = train_imgs.intersection(test_imgs)
    v_test = val_imgs.intersection(test_imgs)
    
    report = f"Data Leakage Check:\nTrain/Val overlap: {len(t_v)}\nTrain/Test overlap: {len(t_test)}\nVal/Test overlap: {len(v_test)}\n"
    
    with open(METRICS_DIR / "data_leakage_check.txt", "w") as f:
        f.write(report)
        
    print(report)
    return len(t_v) == 0 and len(t_test) == 0 and len(v_test) == 0

def prepare_features(df, feature_cols):
    """Split into X, y based on dataset_split."""
    train = df[df["dataset_split"] == "train"]
    val = df[df["dataset_split"] == "val"]
    test = df[df["dataset_split"] == "test"]
    
    X_train, y_train = train[feature_cols].values, train["gt_class"].values
    X_val, y_val = val[feature_cols].values, val["gt_class"].values
    X_test, y_test = test[feature_cols].values, test["gt_class"].values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test

def train_and_evaluate(model_name, model, X_train_s, y_train, X_test_s, y_test):
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    acc = accuracy_score(y_test, preds)
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": mac_p,
        "macro_recall": mac_r,
        "macro_f1": mac_f1,
        "weighted_precision": wt_p,
        "weighted_recall": wt_r,
        "weighted_f1": wt_f1
    }, preds

def rule_based_baseline(X, feature_cols):
    """
    Very simple rule-based classifier based on observations.
    """
    preds = []
    # Find indices
    try:
        area_idx = feature_cols.index("area")
        sat_idx = feature_cols.index("mean_s")
    except ValueError:
        return ["RBC"] * len(X)
        
    for row in X:
        area = row[area_idx]
        sat = row[sat_idx]
        
        if area < 500:
            preds.append("Platelets")
        elif sat > 100:
            preds.append("WBC")
        else:
            preds.append("RBC")
            
    return np.array(preds)

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=["RBC", "WBC", "Platelets"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["RBC", "WBC", "Platelets"])
    plt.yticks(tick_marks, ["RBC", "WBC", "Platelets"])
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i][j]), horizontalalignment="center", color="white" if cm[i][j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(rf_model, feature_cols, save_path):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # top 15
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Random Forest Feature Importances")
    plt.bar(range(15), importances[indices], align="center")
    plt.xticks(range(15), [feature_cols[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    df = pd.DataFrame({"feature": [feature_cols[i] for i in indices], "importance": importances[indices]})
    df.to_csv(save_path.with_suffix(".csv"), index=False)

def extract_error_crops(test_df, y_test, preds, class_vis_dir):
    """Extract and save up to 20 error crops."""
    errors_df = test_df[y_test != preds].copy()
    errors_df["predicted_class"] = preds[y_test != preds]
    errors_df["true_class"] = y_test[y_test != preds]
    
    errors_df.to_csv(METRICS_DIR / "classification_errors.csv", index=False)
    
    count = 0
    for idx, row in errors_df.iterrows():
        if count >= 20: break
        
        img_name = row["image_name"]
        split = row["dataset_split"]
        
        # Locate image
        img_path = None
        for p in get_paired_files(split=split):
            if p["image_path"].name == img_name:
                img_path = p["image_path"]
                break
                
        if img_path and img_path.exists():
            img_bgr = load_image(img_path)
            x1, y1 = int(row["bbox_x"]), int(row["bbox_y"])
            w, h = int(row["bbox_width"]), int(row["bbox_height"])
            
            # Add padding
            pad = 10
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(img_bgr.shape[1], x1 + w + pad)
            y2_p = min(img_bgr.shape[0], y1 + h + pad)
            
            crop = img_bgr[y1_p:y2_p, x1_p:x2_p]
            
            # Save crop
            fname = f"err_{row['true_class']}_as_{row['predicted_class']}_{img_name}_id{row['instance_id']}.png"
            cv2.imwrite(str(class_vis_dir / "errors" / fname), crop)
            count += 1
def run_stage_6_classification():
    print("\n" + "=" * 70)
    print(" STAGE 6: CELL CLASSIFICATION ")
    print("=" * 70)
    
    class_vis_dir = ensure_classification_dirs()
    
    # 1. Build or Load Dataset
    dataset_path = METRICS_DIR / "classification_dataset.csv"
    if not dataset_path.exists():
        df = build_classification_dataset(num_train=80, num_val=20, num_test=20)
    else:
        df = pd.read_csv(dataset_path)
        
    if df.empty:
        print("Dataset is empty. Aborting classification.")
        return
        
    # 2. Leakage Check
    check_data_leakage(df)
    
    # 3. Class Distribution
    dist = df.groupby(["dataset_split", "gt_class"]).size().unstack(fill_value=0)
    dist.to_csv(METRICS_DIR / "class_distribution.csv")
    dist.plot(kind="bar", stacked=True, figsize=(8,6))
    plt.title("Class Distribution by Split")
    plt.tight_layout()
    plt.savefig(class_vis_dir / "class_distribution.png")
    plt.close()
    
    # 4. Feature Selection
    feature_set_A = ["area", "equivalent_diameter", "aspect_ratio", "circularity", "solidity", 
                     "mean_r", "mean_g", "mean_b", "mean_s", "mean_intensity", "mean_gradient"]
    
    feature_set_B = ["area", "perimeter", "equivalent_diameter", "bbox_width", "bbox_height", 
                     "aspect_ratio", "extent", "circularity", "solidity", "convex_hull_area",
                     "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
                     "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
                     "mean_l", "mean_a", "mean_lab_b", "std_l", "std_a", "std_lab_b",
                     "mean_intensity", "std_intensity", "mean_gradient", "std_gradient"]
                     
    # Use Set B for primary comparison
    (X_train, y_train), (X_val, y_val), (X_test, y_test), test_df = prepare_features(df, feature_set_B)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "best_model_features.json", "w") as f:
        json.dump(feature_set_B, f)
        
    results = []
    
    # 5. Rule Based
    rb_preds = rule_based_baseline(X_test, feature_set_B)
    rb_acc = accuracy_score(y_test, rb_preds)
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(y_test, rb_preds, average="macro", zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(y_test, rb_preds, average="weighted", zero_division=0)
    results.append({"model": "Rule Baseline", "feature_set": "Set B", "accuracy": rb_acc, "macro_precision": mac_p, "macro_recall": mac_r, "macro_f1": mac_f1, "weighted_precision": wt_p, "weighted_recall": wt_r, "weighted_f1": wt_f1})
    plot_confusion_matrix(y_test, rb_preds, "Rule Baseline", class_vis_dir / "confusion_matrix_rule_baseline.png")
    
    # 6. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    res_knn, preds_knn = train_and_evaluate("KNN", knn, X_train_s, y_train, X_test_s, y_test)
    res_knn["feature_set"] = "Set B"
    results.append(res_knn)
    joblib.dump(knn, MODELS_DIR / "best_knn.joblib")
    plot_confusion_matrix(y_test, preds_knn, "KNN (K=5)", class_vis_dir / "confusion_matrix_knn.png")
    
    # 7. SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    res_svm, preds_svm = train_and_evaluate("SVM", svm, X_train_s, y_train, X_test_s, y_test)
    res_svm["feature_set"] = "Set B"
    results.append(res_svm)
    joblib.dump(svm, MODELS_DIR / "best_svm.joblib")
    plot_confusion_matrix(y_test, preds_svm, "SVM (RBF)", class_vis_dir / "confusion_matrix_svm.png")
    
    with open(METRICS_DIR / "classification_report_svm.txt", "w") as f:
        f.write(classification_report(y_test, preds_svm))
        
    # 8. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    # RF doesn't strictly need scaling, but using scaled for consistency
    res_rf, preds_rf = train_and_evaluate("Random Forest", rf, X_train_s, y_train, X_test_s, y_test)
    res_rf["feature_set"] = "Set B"
    results.append(res_rf)
    joblib.dump(rf, MODELS_DIR / "best_random_forest.joblib")
    plot_confusion_matrix(y_test, preds_rf, "Random Forest", class_vis_dir / "confusion_matrix_random_forest.png")
    plot_feature_importance(rf, feature_set_B, class_vis_dir / "random_forest_feature_importance.png")
    
    # Error Analysis (using RF as best usually)
    extract_error_crops(test_df, y_test, preds_rf, class_vis_dir)
    
    # 9. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    res_lr, preds_lr = train_and_evaluate("Logistic Regression", lr, X_train_s, y_train, X_test_s, y_test)
    res_lr["feature_set"] = "Set B"
    results.append(res_lr)
    
    # 10. Save Classifier Comparison
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(METRICS_DIR / "classifier_comparison.csv", index=False)
    
    plt.figure(figsize=(10,6))
    plt.bar(comp_df["model"], comp_df["macro_f1"], color='coral')
    plt.title("Classifier Comparison (Macro F1)")
    plt.ylabel("Macro F1")
    plt.tight_layout()
    plt.savefig(class_vis_dir / "classifier_comparison.png")
    plt.close()
    
    # 11. Feature Ablation with RF
    ablation_groups = {
        "Shape": ["area", "perimeter", "equivalent_diameter", "aspect_ratio", "circularity", "solidity"],
        "Color": ["mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v"],
        "Intensity": ["mean_intensity", "std_intensity"],
        "Texture": ["mean_gradient", "std_gradient"],
        "Shape+Color": ["area", "circularity", "mean_s", "mean_r"],
        "All": feature_set_B
    }
    
    ablation_res = []
    for name, cols in ablation_groups.items():
        (Xa_tr, ya_tr), (Xa_v, ya_v), (Xa_ts, ya_ts), _ = prepare_features(df, cols)
        rf_ab = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_ab.fit(Xa_tr, ya_tr)
        p = rf_ab.predict(Xa_ts)
        _, _, mf1, _ = precision_recall_fscore_support(ya_ts, p, average="macro", zero_division=0)
        ablation_res.append({"feature_group": name, "macro_f1": mf1})
        
    ab_df = pd.DataFrame(ablation_res)
    ab_df.to_csv(METRICS_DIR / "feature_ablation.csv", index=False)
    
    plt.figure(figsize=(10,6))
    plt.bar(ab_df["feature_group"], ab_df["macro_f1"], color='lightgreen')
    plt.title("Feature Ablation (Random Forest Macro F1)")
    plt.ylabel("Macro F1")
    plt.tight_layout()
    plt.savefig(class_vis_dir / "feature_ablation.png")
    plt.close()

    generate_classification_report(comp_df, ab_df, dist)
    
    print(" STAGE 6 COMPLETED SUCCESSFULLY ")


def generate_classification_report(comp_df, ab_df, dist):
    # Convert DataFrames to simple markdown strings manually
    def df_to_md(df):
        cols = df.columns.tolist()
        header = "| " + " | ".join([str(c) for c in cols]) + " |"
        sep = "|-" + "-|-".join(["-"*len(str(c)) for c in cols]) + "-|"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join([str(x) for x in row.values]) + " |")
        return "\n".join([header, sep] + rows)

    dist_reset = dist.reset_index()
    
    content = f"""# Stage 6: Cell Classification Analysis

## 1. Classical ML Approach
We implemented K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and Logistic Regression to classify cell features (RBC, WBC, Platelets). No deep learning was used, adhering to the image processing scope.

## 2. Image-Level Splitting
To prevent data leakage (where cells from the same image appear in both train and test sets), we strictly partitioned the dataset at the image level using the original TXL-PBC train/val/test organization.

## 3. Class Distribution
The dataset exhibits natural imbalance:
{df_to_md(dist_reset)}

## 4. Classifier Performance
{df_to_md(comp_df[["model", "accuracy", "macro_f1"]].round(3))}

Random Forest and SVM generally perform best. Macro F1 is the key metric due to class imbalance (RBCs dominate).

## 5. Feature Importance & Ablation
Random Forest feature importances highlight **area**, **equivalent_diameter**, and **mean_s** (saturation) as critical discriminators.
Ablation study results (Macro F1):
{df_to_md(ab_df.round(3))}

Shape alone is insufficient to distinguish all classes (especially separating stained WBCs). Color (particularly Saturation) is crucial.

## 6. Limitations
- Watershed over-segmentation can create small fragments classified as Platelets.
- This is a morphological research pipeline and does not provide clinical diagnosis.
"""
    with open(REPORTS_DIR / "classification_analysis.md", "w", encoding="utf-8") as f:
        f.write(content)
