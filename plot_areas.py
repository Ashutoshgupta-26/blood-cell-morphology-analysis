import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_area_distributions():
    df = pd.read_csv("results/metrics/classification_dataset.csv")
    
    out_dir = Path("results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # We want to see the lower end very clearly, so let's clip at 20000
    df_plot = df[df["area"] < 20000]
    
    for cls, color in zip(["RBC", "WBC", "Platelets"], ["red", "purple", "orange"]):
        subset = df_plot[df_plot["gt_class"] == cls]
        plt.hist(subset["area"], bins=50, alpha=0.5, label=f'{cls} Area', color=color)
        
    plt.axvline(x=133, color='orange', linestyle='--', label='Min Platelet (133)')
    plt.axvline(x=1075, color='red', linestyle='--', label='Min RBC (1075)')
    
    plt.title("Training Area Distributions by Class")
    plt.xlabel("Area (Pixels)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "class_area_distributions.png")
    print("Saved results/debug/class_area_distributions.png")
    
if __name__ == "__main__":
    plot_area_distributions()
