import os
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict


def parse_label_from_filename(filename: str, parser_type: str, emotion_map: Dict[str, str]) -> str:
    """
    Extracts emotion label from filename based on dataset format.
    Supports parser types: 'ravdess', 'crema-d', 'emovo'
    """
    if parser_type == 'ravdess':
        parts = filename.split("-")
        if len(parts) < 3:
            return None
        return emotion_map.get(parts[2])

    elif parser_type == 'crema-d':
        parts = filename.split("_")
        if len(parts) < 3:
            return None
        return emotion_map.get(parts[2].split(".")[0])

    elif parser_type == 'emovo':
    # EMOVO uses a dash format: dis-f2-b1.wav
        parts = filename.split("-")
        if not parts:
            return None
        return emotion_map.get(parts[0].lower())

    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")


def generate_metadata(
    dataset_name: str,
    input_dir: Path,
    output_csv: Path,
    modality: str,
    emotion_map: Dict[str, str],
    parser_type: str,
    selected_emotions: List[str],
    samples_per_class: int = 150
) -> None:
    print(f"[INFO] Generating metadata for {dataset_name}...")

    # Gather all relevant files
    extensions = ["*.mp4", "*.wav"] if modality == "audio_video" else ["*.wav"]
    all_files = []
    for ext in extensions:
        all_files.extend(input_dir.rglob(ext))

    # Process all found files

    # Extract emotion labels
    records = []
    for file in all_files:
        label = parse_label_from_filename(file.name, parser_type, emotion_map)
        if label in selected_emotions:
            records.append({
                "filepath": str(file),
                "label": label,
                "modality": modality
            })

    df = pd.DataFrame(records)
    # Display sample of generated metadata

    if df.empty:
        raise ValueError("No matching emotion labels found. Check filenames or mappings.")

    # Balance dataset
    df_balanced = df.groupby("label").sample(
        n=min(samples_per_class, df['label'].value_counts().min()), random_state=42
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved metadata to: {output_csv} ({len(df_balanced)} samples)")


# === usage ===
if __name__ == "__main__":
    # Configuration for inserted dataset
    generate_metadata(
        dataset_name="RAVDESS",
        input_dir=Path("../data/raw/RAVDESS"),
        output_csv=Path("../data/processed/RAVDESS_metadata.csv"),
        modality="audio_video",
        parser_type="ravdess",
        emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
},
        selected_emotions=["happy", "sad", "angry", "fearful", "disgust", "surprised"]
    )
