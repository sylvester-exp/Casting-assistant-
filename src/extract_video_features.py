import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Optional
import tempfile
import subprocess
import shlex

#  import MoviePy inside functions

# === CONFIGURATION ===
FRAMES_DIR = Path("../data/processed/video_frames/")
OUTPUT_PATH = Path("../features/video_features.csv")
METADATA_CSV = Path("../data/processed/RAVDESS_metadata.csv")  # Can be adjusted


def extract_video_embedding_from_file(input_path: Path, max_frames: int = 64, frame_size: tuple = (224, 224)) -> Optional[np.ndarray]:
    """Sample frames from a video file using MoviePy and return a consistent array.

    Returns a numpy array of shape (N, H, W, 3) with uint8 dtype, where N<=max_frames.
    If no frames can be read, returns None.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    frames = []
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(input_path))
    except Exception:
        clip = None

    if clip is not None:
        try:
            total_duration = clip.duration or 0
            if total_duration <= 0:
                return None

            # Uniformly sample up to max_frames across the duration
            if max_frames <= 0:
                max_frames = 1
            timestamps = np.linspace(0, max(total_duration - 1e-6, 0), num=max_frames)

            for t in timestamps:
                try:
                    frame = clip.get_frame(t)
                except Exception:
                    continue
                if frame is None:
                    continue
                arr = np.asarray(frame)
                frames.append(arr)

            if not frames:
                return None
            batch = np.stack(frames, axis=0).astype(np.uint8)
            return batch
        finally:
            clip.close()

    # MoviePy unavailable or failed: fallback to ffmpeg via imageio-ffmpeg
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as tdir:
            out_tpl = Path(tdir) / "frame_%06d.jpg"
            # Sample at 8 fps but cap frames with -vframes
            cmd = f"{shlex.quote(ffmpeg_exe)} -y -i {shlex.quote(str(input_path))} -vf fps=8 -vframes {int(max_frames)} {shlex.quote(str(out_tpl))}"
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                return None
            imgs = sorted(Path(tdir).glob("frame_*.jpg"))[:max_frames]
            for p in imgs:
                try:
                    arr = imageio.imread(str(p))
                    frames.append(arr)
                except Exception:
                    continue
        if not frames:
            return None
        return np.stack(frames, axis=0).astype(np.uint8)
    except Exception:
        return None


def simple_video_stats(frames: Optional[np.ndarray]) -> dict:
    """Compute lightweight video statistics from frames array (N,H,W,3)."""
    if frames is None or (hasattr(frames, "size") and frames.size == 0):
        return {"vid_mean_brightness": 0.0, "vid_std_brightness": 0.0, "vid_num_frames": 0.0}
    arr = frames.astype(np.float32)
    gray = arr.mean(axis=3)
    mean_b = float(gray.mean())
    std_b = float(gray.std())
    n = float(arr.shape[0])
    # Motion proxy: mean absolute difference between consecutive frames
    if arr.shape[0] > 1:
        diff = np.abs(gray[1:] - gray[:-1])
        motion = float(diff.mean())
    else:
        motion = 0.0
    res =  {
        "vid_mean_brightness": mean_b,
        "vid_std_brightness": std_b,
        "vid_motion": motion,
        "vid_num_frames": n,
    }
    # MediaPipe face landmarks stats (guarded by env + numpy/cv2 compatibility)
    try:
        if os.environ.get("VIDEO_USE_MEDIAPIPE", "0") == "1":
            import numpy as _np
            if int(_np.__version__.split(".")[0]) < 2:
                import mediapipe as mp
                face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
                mouth_aspect = []
                for i in range(min(arr.shape[0], 16)):
                    img = arr[i].astype(np.uint8)
                    results = face_mesh.process(img)
                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0].landmark
                        # Simple mouth openness proxy using selected landmark indices
                        top = lm[13].y; bottom = lm[14].y
                        mouth_aspect.append(abs(bottom - top))
                if mouth_aspect:
                    res["vid_mouth_open_mean"] = float(np.mean(mouth_aspect))
                    res["vid_mouth_open_std"] = float(np.std(mouth_aspect))
    except Exception:
        pass
    return res


def preprocess_video(input_path: Path, output_dir: Path, frame_size: tuple = (224, 224)):
    """Extracts and saves resized frames from .mp4 video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use MoviePy to iterate frames and save a subset to disk for training pipelines
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(str(input_path))
    try:
        total_duration = clip.duration or 0
        if total_duration <= 0:
            return
        timestamps = np.linspace(0, max(total_duration - 1e-6, 0), num=128)
        frame_idx = 0
        for t in timestamps:
            try:
                frame = clip.get_frame(t)
            except Exception:
                continue
            frame_path = output_dir / f"frame_{frame_idx:04d}.jpg"
            # Lazy import to avoid global cv2 dependency in dashboard path
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(frame_path), frame)
            except Exception:
                pass
            frame_idx += 1
    finally:
        clip.close()


def generate_frames_from_metadata(metadata_path: Path, output_root: Path):
    """Generates video frames for audio_video files in a metadata CSV."""
    df = pd.read_csv(metadata_path)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Generating video frames"):
        filepath = Path(row.filepath)
        modality = getattr(row, "modality", "audio_video")
        label = row.label
        stem = filepath.stem

        if modality == "audio_video" and filepath.suffix == ".mp4":
            output_video_path = FRAMES_DIR / label / stem
            preprocess_video(filepath, output_video_path)


def process_all_videos(frames_root: Path) -> pd.DataFrame:
    """Extracts face embeddings from all pre-extracted video frames."""
    records = []

    for label_dir in sorted(frames_root.iterdir()):
        if not label_dir.is_dir():
            continue

        for video_folder in tqdm(sorted(label_dir.iterdir()), desc=f"Embedding: {label_dir.name}"):
            if not video_folder.is_dir():
                continue

            # Backwards compatibility: keep using pre-extracted frames with face_recognition removed.
            # Here we simply compute a mean RGB per frame folder as a placeholder embedding to avoid breaking training.
            frame_files = sorted(video_folder.glob("*.jpg"))
            if not frame_files:
                embedding = None
            else:
                import imageio.v2 as imageio
                pixels = []
                for f in frame_files:
                    try:
                        arr = imageio.imread(str(f))
                        pixels.append(arr.mean(axis=(0, 1)))
                    except Exception:
                        continue
                if pixels:
                    embedding = np.mean(np.stack(pixels, axis=0), axis=0)
                else:
                    embedding = None
            if embedding is not None:
                record = {
                    f"vfeat_{i+1}": val for i, val in enumerate(embedding)
                }
                record["label"] = label_dir.name
                record["video_id"] = video_folder.name
                records.append(record)


    return pd.DataFrame(records)


def main(regenerate_frames: bool):
    if regenerate_frames:
        print(f"[INFO] Regenerating video frames from: {METADATA_CSV}")
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        generate_frames_from_metadata(METADATA_CSV, FRAMES_DIR)

    print(f"[INFO] Extracting video embeddings from: {FRAMES_DIR}")
    df = process_all_videos(FRAMES_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Saved video features to: {OUTPUT_PATH} ({len(df)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video features with optional frame regeneration.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate video frames from metadata before extraction")
    args = parser.parse_args()
    main(regenerate_frames=args.regenerate)
