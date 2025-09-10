from pathlib import Path
from src.extract_audio_features import extract_features as A
from src.extract_video_features import extract_video_embedding_from_file as V
p = Path("tests/sample.mov")  # set to a real short clip
print("Audio:", A(p))
emb = V(p)
print("Video frames/feat len:", len(emb))
