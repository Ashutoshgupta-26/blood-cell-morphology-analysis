"""
Stage 10: Streamlit Interactive Dashboard
A polished interface wrapping the complete Stage 1-9 Classical Image Processing Pipeline
for Blood Cell Detection, Counting, and Morphological Analysis.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import time

# Internal project imports
from src.counting import load_classification_pipeline, analyze_blood_smear, generate_counting_metrics
from src.morphological_analysis import define_feature_groups, detect_outliers, generate_visualizations
from config import VISUALIZATIONS_DIR

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Blood Cell Morphological Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .disclaimer {
        color: #e63946;
        font-size: 0.85em;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        padding: 10px;
        border: 1px solid #e63946;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching Core Models ---
@st.cache_resource
def get_model_pipeline():
    try:
        return load_classification_pipeline()
    except FileNotFoundError:
        st.error("Trained classification model not found. Please run Stage 6/9 model training first.")
        st.stop()

scaler, model, feature_cols = get_model_pipeline()

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/Red_White_Blood_cells.jpg", use_container_width=True, caption="Microscopy Reference")
    st.title("Project Information")
    st.markdown("### Blood Cell Detection, Counting and Morphological Analysis")
    st.markdown("**Dataset**: TXL-PBC\n**Classes**: RBC, WBC, Platelet")
    st.markdown("### Pipeline Methods")
    st.markdown("- Classical image preprocessing\n- Threshold-based segmentation\n- Morphological operations\n- Distance transform & Watershed\n- Hand-crafted image features\n- Classical SVM classification\n- Z-score & IQR Outliers")
    st.markdown("---")
    st.info("The frozen Stage 9 pipeline is deployed here without modification.")

# --- Helper Functions ---
def draw_annotated_image(img_bgr, labels, preds_df, selected_classes):
    """Draw boundaries selectively based on user filter."""
    vis_img = img_bgr.copy()
    class_colors = {"RBC": (50, 50, 220), "WBC": (255, 50, 150), "Platelets": (0, 200, 255)}
    
    for _, row in preds_df.iterrows():
        pred_class = row["predicted_class"]
        if "All" not in selected_classes and pred_class not in selected_classes:
            continue
            
        inst_id = int(row["instance_id"])
        color = class_colors.get(pred_class, (255, 255, 255))
        
        mask = np.uint8(labels == inst_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, 2)
        
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{'PLT' if pred_class=='Platelets' else pred_class} #{inst_id}"
                cv2.putText(vis_img, label, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

def draw_outlier_overlay(img_bgr, labels, df_outliers):
    vis_img = img_bgr.copy()
    normal_color = (150, 150, 150)  # Gray for normal
    outlier_color = (0, 0, 255)     # Bright Red for outliers
    
    for _, row in df_outliers.iterrows():
        inst_id = int(row["instance_id"])
        is_out = row["high_confidence_morphological_outlier"] == 1
        
        color = outlier_color if is_out else normal_color
        thick = 2 if is_out else 1
        
        mask = np.uint8(labels == inst_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, thick)
        
    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

# --- Main UI ---
st.title("Blood Cell Detection & Morphological Analysis")
st.markdown("##### Classical Image Processing Pipeline for Blood Cell Segmentation, Classification, Counting and Morphological Analysis")

uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # Save a temporary path for the analysis function to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=f"Original Image: {uploaded_file.name} ({img_bgr.shape[1]}x{img_bgr.shape[0]})", width=600)
    
    if st.button("Analyze Image", type="primary"):
        start_time = time.time()
        
        with st.status("Executing End-to-End Pipeline...", expanded=True) as status:
            st.write("1. Preprocessing image...")
            time.sleep(0.5)
            st.write("2. Segmenting cells (Otsu)...")
            st.write("3. Separating touching cells (Watershed)...")
            st.write("4. Extracting morphological features...")
            st.write("5. Classifying cells (SVM)...")
            
            # Execute Pipeline
            result = analyze_blood_smear(tmp_path, scaler, model, feature_cols)
            preds_df = result["predictions_df"]
            counts = result["counts"]
            
            if preds_df.empty:
                status.update(label="Analysis failed: No cells detected.", state="error")
                st.stop()
                
            st.write("6. Detecting morphological outliers...")
            groups = define_feature_groups()
            df_out, reasons_df = detect_outliers(preds_df, groups)
            
            status.update(label="Analysis complete!", state="complete")
            
        proc_time = time.time() - start_time
        st.success(f"Processing time: {proc_time:.2f} seconds")
        
        # --- TOP LEVEL RESULTS ---
        st.markdown("---")
        st.subheader("RESULTS")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Cells", counts["Total"])
        col2.metric("RBC", counts["RBC"])
        col3.metric("WBC", counts["WBC"])
        col4.metric("Platelets", counts["Platelets"])
        
        st.markdown("<div class='disclaimer'>Educational/research image analysis only. The detected cell counts are counts of image objects and should not be interpreted as clinical blood counts. Morphological outliers are not medical diagnoses.</div>", unsafe_allow_html=True)
        
        # --- TABS FOR ORGANIZED VIEW ---
        tab1, tab2, tab3, tab4 = st.tabs(["Final Analyzed Image", "Cell Details", "Morphology & Distributions", "Outliers"])
        
        with tab1:
            st.markdown("### Interactive Overlay")
            class_filter = st.multiselect("Show cell classes:", options=["All", "RBC", "WBC", "Platelet"], default=["All"])
            
            # Map Platelet -> Platelets
            mapped_filter = []
            for c in class_filter:
                if c == "Platelet": mapped_filter.append("Platelets")
                else: mapped_filter.append(c)
                
            annotated_img = draw_annotated_image(img_bgr, result["labels"], preds_df, mapped_filter)
            st.image(annotated_img, use_container_width=True, caption="Detected Boundaries & Classes")
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig_bar, ax_bar = plt.subplots(figsize=(5,3))
                
                # Get actual counts from predictions dataframe
                actual_counts = preds_df["predicted_class"].value_counts()
                
                # Map to consistent colors
                color_map = {"RBC": "red", "WBC": "purple", "Platelets": "orange"}
                
                present_classes = []
                present_values = []
                present_colors = []
                
                for cls in ["RBC", "WBC", "Platelets"]:
                    if cls in actual_counts and actual_counts[cls] > 0:
                        present_classes.append(cls)
                        present_values.append(actual_counts[cls])
                        present_colors.append(color_map[cls])
                
                if present_classes:
                    ax_bar.bar(present_classes, present_values, color=present_colors)
                    ax_bar.set_title("Cell Distribution")
                    st.pyplot(fig_bar)
                else:
                    st.info("No cells detected for distribution chart.")
                    
            with col_b:
                if len(present_classes) > 0:
                    fig_pie, ax_pie = plt.subplots(figsize=(5,3))
                    ax_pie.pie(present_values, labels=present_classes, colors=present_colors, autopct='%1.1f%%')
                    ax_pie.set_title("Class Percentages")
                    st.pyplot(fig_pie)
                    
            st.download_button("Download Image", data=cv2.imencode('.png', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name="annotated_smear.png", mime="image/png")
            
        with tab2:
            st.markdown("### Detected Cells")
            display_cols = ["instance_id", "predicted_class", "prediction_confidence", "area", "equivalent_diameter", "circularity", "aspect_ratio", "solidity", "mean_intensity", "bbox_x", "bbox_y"]
            st.dataframe(df_out[display_cols].sort_values("instance_id"), use_container_width=True)
            
            with st.expander("Show all extracted features"):
                st.dataframe(df_out, use_container_width=True)
                
            st.markdown("### Cell Inspector")
            selected_id = st.selectbox("Select Cell ID to inspect:", options=df_out["instance_id"].values)
            if selected_id:
                row = df_out[df_out["instance_id"] == selected_id].iloc[0]
                x1, y1 = int(row["bbox_x"]), int(row["bbox_y"])
                w, h = int(row["bbox_width"]), int(row["bbox_height"])
                pad = 10
                crop = img_bgr[max(0, y1-pad):min(img_bgr.shape[0], y1+h+pad), max(0, x1-pad):min(img_bgr.shape[1], x1+w+pad)]
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"ID: {selected_id} | Class: {row['predicted_class']}", width=150)
                with c2:
                    st.write(f"**Area**: {row['area']:.2f} px")
                    st.write(f"**Circularity**: {row['circularity']:.3f}")
                    st.write(f"**Solidity**: {row['solidity']:.3f}")
                    st.write(f"**Color (LAB-b)**: {row['mean_lab_b']:.2f}")
                    st.write(f"**Outlier Score**: {row['outlier_score']}")
            
            st.download_button("Download Feature CSV", df_out.to_csv(index=False), "cell_features.csv", "text/csv")

        with tab3:
            st.markdown("### Size Distribution")
            fig_area, ax_area = plt.subplots(figsize=(6,4))
            for cls in ["RBC", "WBC", "Platelets"]:
                sub = df_out[df_out["predicted_class"] == cls]
                if not sub.empty:
                    ax_area.hist(sub["area"], bins=15, alpha=0.5, label=cls)
            ax_area.set_title("Area Distribution by Class")
            ax_area.legend()
            st.pyplot(fig_area)
            
            st.markdown("### Shape Distribution")
            fig_circ, ax_circ = plt.subplots(figsize=(6,4))
            for cls in ["RBC", "WBC", "Platelets"]:
                sub = df_out[df_out["predicted_class"] == cls]
                if not sub.empty:
                    ax_circ.hist(sub["circularity"], bins=15, alpha=0.5, label=cls)
            ax_circ.set_title("Circularity Distribution by Class")
            ax_circ.legend()
            st.pyplot(fig_circ)

        with tab4:
            high_conf = df_out[df_out["high_confidence_morphological_outlier"] == 1]
            st.markdown(f"### Morphological Outliers: {len(high_conf)}")
            st.warning("Outliers represent image-based deviations from the observed class distribution. They are not medical diagnoses.")
            
            outlier_img = draw_outlier_overlay(img_bgr, result["labels"], df_out)
            st.image(outlier_img, use_container_width=True, caption="Outliers Highlighted (Red)")
            
            if not high_conf.empty:
                st.dataframe(high_conf[["instance_id", "predicted_class", "area", "circularity", "outlier_score"]], use_container_width=True)
                
                st.markdown("#### Outlier Inspector")
                out_id = st.selectbox("Inspect Outlier:", options=high_conf["instance_id"].values)
                if out_id:
                    row = high_conf[high_conf["instance_id"] == out_id].iloc[0]
                    reasons = reasons_df[reasons_df["instance_id"] == out_id]
                    
                    x1, y1 = int(row["bbox_x"]), int(row["bbox_y"])
                    w, h = int(row["bbox_width"]), int(row["bbox_height"])
                    pad = 10
                    crop = img_bgr[max(0, y1-pad):min(img_bgr.shape[0], y1+h+pad), max(0, x1-pad):min(img_bgr.shape[1], x1+w+pad)]
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"ID: {out_id} | Class: {row['predicted_class']}", width=150)
                    with c2:
                        st.write("**Reasons for Flagging:**")
                        st.dataframe(reasons[["feature", "method", "value", "reference_mean"]])
                
                st.download_button("Download Outlier CSV", high_conf.to_csv(index=False), "outliers.csv", "text/csv")
            else:
                st.success("No high-confidence morphological outliers detected in this image.")
                
        # --- GENERATE SUMMARY REPORT ---
        report = f"""Blood Cell Analysis Summary
---------------------------
Image: {uploaded_file.name}
Resolution: {img_bgr.shape[1]}x{img_bgr.shape[0]}

Total Detected Cells: {counts['Total']}
- RBC: {counts['RBC']}
- WBC: {counts['WBC']}
- Platelets: {counts['Platelets']}

Morphological Outliers: {len(high_conf)}
Most Common Cell: {max(counts, key=counts.get) if counts['Total'] > 0 else 'None'}

Average RBC Area: {df_out[df_out['predicted_class'] == 'RBC']['area'].mean():.2f} px
Average WBC Area: {df_out[df_out['predicted_class'] == 'WBC']['area'].mean():.2f} px
Average RBC Circularity: {df_out[df_out['predicted_class'] == 'RBC']['circularity'].mean():.3f}

DISCLAIMER: This is an image-analysis result and is not a clinical blood count or medical diagnosis.
"""
        st.download_button("Generate & Download Analysis Summary", report, "analysis_summary.txt", "text/plain", type="primary")

