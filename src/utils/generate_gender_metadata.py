
import pandas as pd
from pathlib import Path

def extract_gender(actor_id):
    try:
        return "male" if int(actor_id) % 2 != 0 else "female"
    except:
        return "unknown"

def generate_metadata(input_path, output_path):
    df = pd.read_csv(input_path)
    df["filename"] = df["filepath"].apply(lambda x: Path(x).name)
    df["actor_id"] = df["filename"].apply(lambda x: x.split("-")[-1].split(".")[0])
    df["gender"] = df["actor_id"].apply(extract_gender)
    df[["filename", "gender"]].to_csv(output_path, index=False)
    print(f"[INFO] Metadata with gender saved to: {output_path}")

if __name__ == "__main__":
    generate_metadata(
        input_path="data/processed/RAVDESS_metadata.csv",
        output_path="features/metadata.csv"
    )
