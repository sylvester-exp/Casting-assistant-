import cv2
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from pathlib import Path


# === CONFIGURATION ===
SAMPLE_RATE = 16000
FRAME_SIZE = (224, 224)
AUDIO_OUT_DIR = Path("../data/processed/audio_cleaned")
VIDEO_OUT_DIR = Path("../data/processed/video_frames")


# === AUDIO PROCESSING ===
def preprocess_audio(input_path: Path, output_path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Loads, resamples, and saves audio to mono WAV format."""
    try:
        audio, _ = librosa.load(str(input_path), sr=sample_rate, mono=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate)
    except Exception as e:
        print(f"[ERROR] Audio processing failed for {input_path}: {e}")


# === VIDEO PROCESSING ===
def preprocess_video(input_path: Path, output_dir: Path, frame_size: tuple = FRAME_SIZE) -> None:
    """Extracts resized frames from a video and saves them as JPGs."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(input_path))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, frame_size)
            frame_path = output_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), resized)
            frame_idx += 1

        cap.release()
    except Exception as e:
        print(f"[ERROR] Video processing failed for {input_path}: {e}")


# === MAIN PIPELINE ===
def process_metadata_file(metadata_path: Path) -> None:
    """Processes audio and/or video from a given metadata CSV."""
    print(f"[INFO] Processing dataset: {metadata_path.name}")
    df = pd.read_csv(metadata_path)

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        input_path = Path(row.filepath)
        label = row.label
        modality = row.modality
        stem = input_path.stem

        if modality in {"audio", "audio_video"} and input_path.suffix in {".wav", ".mp4"}:
            output_audio = AUDIO_OUT_DIR / label / f"{stem}.wav"
            preprocess_audio(input_path, output_audio)

        if modality == "audio_video" and input_path.suffix == ".mp4":
            output_frames_dir = VIDEO_OUT_DIR / label / stem
            preprocess_video(input_path, output_frames_dir)

    print(f"[DONE] Finished: {metadata_path.name}")


def main():
    metadata_paths = [
        Path("../data/processed/RAVDESS_metadata.csv"),
        Path("../data/processed/CREMA-D_metadata.csv"),
        Path("../data/processed/EMOVO_metadata.csv")
    ]

    for path in metadata_paths:
        process_metadata_file(path)


if __name__ == "__main__":
    main()
