import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_group_f1(preds_path, metadata_path, group_col="gender", output_dir="outputs"):
    # Load data
    preds_df = pd.read_csv(preds_path)
    meta_df = pd.read_csv(metadata_path)

    # Merge on filename
    df = preds_df.merge(meta_df, on="filename")

    # Grouped F1 calculation
    scores = df.groupby(group_col).apply(lambda g: f1_score(g["label"], g["predicted"], average="weighted")).reset_index()
    scores.columns = [group_col, "f1_score"]

    # Save as CSV
    scores.to_csv(Path(output_dir) / f"fairness_f1_by_{group_col}.csv", index=False)

    # Plot
    sns.barplot(data=scores, x=group_col, y="f1_score")
    plt.title(f"F1 Score by {group_col.capitalize()}")
    plt.ylabel("F1 Score")
    plt.xlabel(group_col.capitalize())
    plt.tight_layout()
    plot_path = Path(output_dir) / f"fairness_f1_by_{group_col}.png"
    plt.savefig(plot_path)
    print(f"[INFO] Fairness plot saved to {plot_path}")

if __name__ == "__main__":
    evaluate_group_f1(
        preds_path="outputs/predictions.csv",
        metadata_path="features/fused_features.csv",  # <- You must provide this
        group_col="gender"
    )
