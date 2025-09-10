import json
from pathlib import Path

def _load_role_profiles() -> dict:
    """Load role profiles from common locations; return {} if not found."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "config" / "role_profiles.json",      # project-root/config/
        Path.cwd() / "config" / "role_profiles.json",       # current working dir
    ]
    for p in candidates:
        try:
            if p.exists():
                with open(p, "r") as f:
                    return json.load(f)
        except Exception:
            continue
    return {}

_ROLE_PROFILES = _load_role_profiles()

def match_role(emotion: str) -> str:
    profiles = _ROLE_PROFILES or {}
    for role, profile in profiles.items():
        if emotion in profile.get("dominant_emotions", []):
            return role
    return "unknown"

if __name__ == "__main__":
    # Optional: batch apply on fused file if invoked directly
    import pandas as pd
    FUSED_PATH = Path("../features/fused_features.csv")
    df = pd.read_csv(FUSED_PATH)
    label_map = {
        0: "angry",
        1: "disgust",
        2: "fearful",
        3: "happy",
        4: "sad",
        5: "surprised"
    }
    df["predicted_emotion"] = df["emotion_id"].map(label_map)
    df["matched_role"] = df["predicted_emotion"].apply(match_role)
    OUTPUT_PATH = Path("../outputs/role_matches.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Saved role matches to: {OUTPUT_PATH}")
