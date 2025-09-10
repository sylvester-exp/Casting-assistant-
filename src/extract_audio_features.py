import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from typing import Union

#import MoviePy inside functions to avoid hard failures on module import
import subprocess
import shlex


# === CONFIG ===
AUDIO_DIR = Path("../data/processed/audio_cleaned/")
OUTPUT_PATH = Path("../features/audio_features.csv")
SAMPLE_RATE = 16000
N_MFCC = 13


def _demux_to_wav(input_path: Path, sample_rate: int) -> Path:
    """Always convert input (audio/video) to a temporary WAV file."""
    suffix = input_path.suffix.lower()
    tmp = NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = Path(tmp.name)
    tmp.close()

    # Try AudioFileClip for audio containers; fall back to VideoFileClip for video
    clip = None
    try:
        from moviepy.editor import AudioFileClip
        clip = AudioFileClip(str(input_path))
    except Exception:
        clip = None

    if clip is None:
        try:
            from moviepy.editor import VideoFileClip
            vclip = VideoFileClip(str(input_path))
            if vclip.audio is None:
                vclip.close()
                raise ValueError("No audio stream found in video container.")
            clip = vclip.audio
        except Exception:
            # MoviePy failed. Fallback to calling ffmpeg directly via imageio-ffmpeg.
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = f"{shlex.quote(ffmpeg_exe)} -y -i {shlex.quote(str(input_path))} -vn -ac 1 -ar {int(sample_rate)} -acodec pcm_s16le {shlex.quote(str(tmp_path))}"
                proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[:500])
                return tmp_path
            except Exception as exc2:
                raise RuntimeError(f"Failed to open media file for audio demux: {input_path}") from exc2

    try:
        clip.write_audiofile(
            str(tmp_path),
            fps=sample_rate,
            nbytes=2,
            codec="pcm_s16le",
            verbose=False,
            logger=None,
        )
    finally:
        clip.close()

    return tmp_path


def extract_features(
    file_path: Union[Path, str],
    sample_rate: int = SAMPLE_RATE,
    offset_sec: float | None = None,
    duration_sec: float | None = None,
    use_vad: bool = True,
) -> dict:
    """Extract MFCCs, delta-MFCCs, pitch, energy and spectral features.

    Ensures non-WAV inputs (including video containers) are demuxed to WAV using
    MoviePy/imageio-ffmpeg before calling librosa.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    wav_path = None
    try:
        if path.suffix.lower() == ".wav":
            load_target = path
        else:
            wav_path = _demux_to_wav(path, sample_rate)
            load_target = wav_path

        y, sr = librosa.load(
            str(load_target), sr=sample_rate, mono=True,
            offset=offset_sec or 0.0, duration=duration_sec
        )

        if y is None or y.size == 0:
            raise ValueError("Empty audio after demux and load.")

        # Voice activity detection (trim non-silent regions)
        if use_vad:
            try:
                intervals = librosa.effects.split(y, top_db=20)
                if intervals is not None and len(intervals) > 0:
                    # Concatenate voiced segments; cap to 30 seconds for safety
                    voiced = np.concatenate([y[s:e] for s, e in intervals])
                    if voiced.size > 0:
                        y = voiced[: sr * 30]
            except Exception:
                pass

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Delta MFCCs
        delta = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta, axis=1)

        # Pitch (via librosa pyin)
        pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        pitch_mean = float(np.nanmean(pitch)) if pitch is not None else 0.0

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))

        # Additional spectral features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        # Tonnetz requires harmonic component
        try:
            y_harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
        except Exception:
            tonnetz_mean = np.zeros(6, dtype=float)

        # Combine all features
        features = {
            f"mfcc_{i+1}": float(val) for i, val in enumerate(mfccs_mean)
        }
        features.update({
            f"delta_mfcc_{i+1}": float(val) for i, val in enumerate(delta_mean)
        })
        features["pitch"] = pitch_mean
        features["rms_energy"] = rms_mean
        features["spectral_centroid"] = spec_centroid
        features["spectral_bandwidth"] = spec_bw
        features["spectral_rolloff"] = spec_rolloff
        features["zero_crossing_rate"] = zcr
        for i, val in enumerate(chroma_mean):
            features[f"chroma_{i+1}"] = float(val)
        for i, val in enumerate(spec_contrast_mean):
            features[f"spec_contrast_{i+1}"] = float(val)
        for i, val in enumerate(tonnetz_mean):
            features[f"tonnetz_{i+1}"] = float(val)
        features["duration_sec"] = float(len(y) / sr)
        features["sample_rate"] = int(sr)

        return features
    finally:
        # Clean up temp wav if we created one
        if wav_path is not None:
            try:
                os.remove(wav_path)
            except Exception:
                pass


def process_dataset(audio_dir: Path) -> pd.DataFrame:
    records = []
    for label_dir in sorted(audio_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        for wav_file in tqdm(label_dir.glob("*.wav"), desc=f"Processing {label_dir.name}"):
            features = extract_features(wav_file)
            if features:
                features["label"] = label_dir.name
                features["filename"] = wav_file.name
                records.append(features)

    return pd.DataFrame(records)


def main():
    print(f"[INFO] Extracting audio features from: {AUDIO_DIR}")
    df = process_dataset(AUDIO_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Saved features to: {OUTPUT_PATH} ({len(df)} samples)")


if __name__ == "__main__":
    main()
