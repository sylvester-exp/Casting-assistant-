import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Local feature extractor (audio)
from src.extract_audio_features import extract_features as extract_audio_features
from src.extract_video_features import extract_video_embedding_from_file, simple_video_stats

# === File paths (relative to project root)
CREMA_PATH = Path("data/processed/CREMA-D_metadata.csv")
EMOVO_PATH = Path("data/processed/EMOVO_metadata.csv")
RAVDESS_PATH = Path("data/processed/RAVDESS_metadata.csv")
OUTPUT_PATH = Path("features/fused_features.csv")


def load_and_prepare(path: Path, modality: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names for consistent processing

    # Flexible column renaming
    rename_success = False
    for col in df.columns:
        lowered = col.lower()
        if "filename" in lowered or "file" in lowered:
            # Keep original file path
            df["filepath"] = df[col].astype(str)
            # Derive a stable ID from the stem
            df["video_id"] = df[col].astype(str).apply(lambda x: Path(str(x)).stem)
            rename_success = True
            break

    if not rename_success:
        raise ValueError(
            f"[ERROR] Could not find a column like 'filename' or 'file' in {path.name}. "
            f"Found columns: {df.columns.tolist()}"
        )

    #  operate on video_id
    df["video_id"] = df["video_id"].astype(str).str.replace(".wav", "", regex=False)
    df["video_id"] = df["video_id"].astype(str).str.replace(".mp4", "", regex=False)

    df["modality"] = modality
    df["has_audio"] = True
    df["has_video"] = modality == "audio_video"
    return df



def main():
    datasets = []
    for path, modality in [
        (CREMA_PATH, "audio_only"),
        (EMOVO_PATH, "audio_only"),
        (RAVDESS_PATH, "audio_video"),
    ]:
        try:
            if not Path(path).exists():
                print(f"[WARN] Skipping missing dataset: {path}")
                continue
            df = load_and_prepare(path, modality)
            datasets.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}. Skipping.")

    if not datasets:
        raise FileNotFoundError("No datasets found in data/processed/. Provide at least one metadata CSV.")

    # Standardise label format
    for df in datasets:
        df["label"] = df["label"].str.lower().str.strip()

    # Combine them
    full_df = pd.concat(datasets, ignore_index=True)

    # Balanced per-class sampling (before any head() truncation)
    per_class = os.environ.get("FUSE_PER_CLASS")
    if per_class:
        try:
            k = int(per_class)
            full_df = (
                full_df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), k), random_state=42))
                .reset_index(drop=True)
            )
            print(f"[INFO] Balanced sampling up to {k} rows per label (FUSE_PER_CLASS)")
        except Exception as e:
            print(f"[WARN] FUSE_PER_CLASS ignored: {e}")

    # Optional final cap for faster development
    max_rows = os.environ.get("FUSE_MAX_ROWS")
    if max_rows:
        try:
            n = int(max_rows)
            if len(full_df) > n:
                full_df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
            print(f"[INFO] Using at most {n} rows for feature fusion (FUSE_MAX_ROWS)")
        except Exception:
            pass
    print(f"[INFO] Combined dataset shape: {full_df.shape}")

    # Helper to resolve file paths from metadata that may point to non-existent raw dirs
    def _resolve_media_path(p: Path) -> Path:
        if p.exists():
            return p
        # Try project-root relative
        if (Path('.') / p).exists():
            return Path('.') / p
        # Try data/processed with basename
        basename = p.name
        candidate = Path('data/processed') / basename
        if candidate.exists():
            return candidate
        # Try data-raw with basename or with replaced prefix
        candidate = Path('data-raw') / basename
        if candidate.exists():
            return candidate
        # Replace common prefixes like ../data/raw or data/raw with data-raw
        try:
            s = str(p)
            s = s.replace('../data/raw/', 'data-raw/').replace('data/raw/', 'data-raw/')
            rp = Path(s)
            if rp.exists():
                return rp
        except Exception:
            pass
        # Try any match under data/
        try:
            matches = list(Path('data').rglob(basename))
            if matches:
                return matches[0]
            matches = list(Path('data-raw').rglob(basename))
            if matches:
                return matches[0]
        except Exception:
            pass
        return p  # will be handled as missing later

    # === Compute audio/video features per file ===
    records = []
    seg_len_env = os.environ.get("FUSE_SEGMENT_SEC")
    try:
        seg_len = float(seg_len_env) if seg_len_env else None
    except Exception:
        seg_len = None
    for row in tqdm(full_df.itertuples(index=False), total=len(full_df), desc="Audio features"):
        try:
            fp = _resolve_media_path(Path(getattr(row, "filepath", "")))
            if not fp.exists():
                # Skip missing files gracefully
                continue
            # Speed: optionally compute features on a short segment
            if seg_len and seg_len > 0:
                # Try multiple segments and aggregate for robustness
                offsets = [0.0, seg_len, 2 * seg_len]
                seg_feats = []
                for off in offsets:
                    try:
                        f = extract_audio_features(fp, offset_sec=off, duration_sec=seg_len)
                        if f:
                            seg_feats.append(f)
                    except Exception:
                        continue
                if seg_feats:
                    # Aggregate mean and std for numeric keys
                    keys = seg_feats[0].keys()
                    agg = {}
                    for k in keys:
                        try:
                            vals = [float(s[k]) for s in seg_feats if k in s]
                            if vals:
                                agg[f"{k}_mean"] = float(np.mean(vals))
                                agg[f"{k}_std"] = float(np.std(vals))
                        except Exception:
                            continue
                    feats = agg if agg else seg_feats[0]
                else:
                    feats = extract_audio_features(fp)
            else:
                feats = extract_audio_features(fp)
            if feats is None:
                continue
            feats_record = {**feats}
            feats_record["video_id"] = str(getattr(row, "video_id"))
            feats_record["label"] = str(getattr(row, "label")).lower().strip()
            # Video stats if video present
            try:
                if str(getattr(row, "modality", "audio_only")) == "audio_video" and fp.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
                    frames = extract_video_embedding_from_file(fp, max_frames=32)
                    vstats = simple_video_stats(frames)
                    feats_record.update(vstats)
            except Exception:
                pass
            records.append(feats_record)
        except Exception as e:
            print(f"[WARN] Audio feature failure for {getattr(row, 'filepath', '')}: {e}")

    if not records:
        raise RuntimeError("No audio features could be computed. Metadata paths do not resolve to files in ./data. Put media under ./data and rerun.")

    audio_df = pd.DataFrame(records)
    print(f"[INFO] Audio features shape: {audio_df.shape}")

    # Merge audio features back on video_id and label
    merged = pd.merge(full_df, audio_df, on=["video_id", "label"], how="inner")
    print(f"[INFO] Merged fused dataset shape: {merged.shape}")

    # Encode emotion lavels
    le = LabelEncoder()
    merged["emotion_id"] = le.fit_transform(merged["label"])
    print("[INFO] Encoded labels:", dict(zip(le.classes_, le.transform(le.classes_))))

   
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Saved fused dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
