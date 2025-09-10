import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
from tempfile import NamedTemporaryFile
from datetime import datetime
import matplotlib.pyplot as plt
import traceback
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Must be the first Streamlit command
st.set_page_config(page_title="AI Casting Assistant", layout="centered")

# === Define ColumnSelector class for model loading ===
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]

# === Project module imports ===
from src.extract_audio_features import extract_features as extract_audio_features
from src.extract_video_features import extract_video_embedding_from_file as extract_video_features
from src.extract_video_features import simple_video_stats
from src.train_classifier import load_trained_model
from src.match_roles import match_role

# === Load classifier model
try:
    bundle = load_trained_model()
    model = bundle.get("pipeline")
    expected_cols = bundle.get("feature_names", [])
    label_classes = bundle.get("label_classes", [])
    class_thresholds = bundle.get("thresholds", {})
    shap_background = bundle.get("shap_background")
    model_base = bundle.get("model_base", "")
    threshold_min = bundle.get("threshold_min", 0.35)
except Exception as e:
    model = None
    st.warning(f"Model not available: {e}")

# === Label map
label_map = {
    0: "angry", 1: "disgust", 2: "fearful",
    3: "happy", 4: "sad", 5: "surprised"
}

#  Streamlit layout
st.title("🎭 AI Casting Assistant")
st.markdown("Upload an audition clip to analyse emotion and match the actor to the most suitable role.")

#  Upload interface 
uploaded_file = st.file_uploader(
    "📹 Upload Audition Video (.mp4)", type=["mp4", "mov", "webm", "mkv", "wav", "m4a", "mp3"]
)

# Only proceeds if a file was actually uploaded 
if uploaded_file is None:
    st.warning("Please upload a video file to continue.")
    st.stop()

# Read the bytes once
data = uploaded_file.read()

# Show the video in the app
st.video(data)

# === Save uploaded file to disk ===
ext = Path(uploaded_file.name).suffix  # “.mp4”, “.mov”, etc.
with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp.write(data)
    tmp.flush()
    tmp.close()
    temp_path = tmp.name


#  Feature extraction 
st.info("🔍 Extracting audio and video features…")

audio_feats, video_feats = None, None
vstats = {}
debug = {"temp_path": temp_path}

# Audio
try:
    audio_feats = extract_audio_features(temp_path)
    if isinstance(audio_feats, dict):
        debug["audio_duration_sec"] = audio_feats.get("duration_sec")
        debug["audio_sample_rate"] = audio_feats.get("sample_rate")
except Exception as e:
    st.warning(f"Audio feature extraction failed: {e}")
    st.exception(e)

# Video
try:
    video_feats = extract_video_features(temp_path)
    if video_feats is not None:
        # numpy array of frames
        debug["num_video_frames"] = getattr(video_feats, "shape", None)[0] if hasattr(video_feats, "shape") else len(video_feats)
        vstats = simple_video_stats(video_feats)
        debug.update(vstats)
except Exception as e:
    st.warning(f"Video feature extraction failed: {e}")
    st.exception(e)

#  Error handling 
if audio_feats is None or video_feats is None:
    st.error("Audio or video feature extraction failed.")
    with st.expander("🔧 Debug details"):
        st.write(debug)
    st.stop()

def _row_from_audio_feats(audio_dict, expected, extra_dict=None):
    if not expected:
        return audio_dict
    row_local = {c: 0.0 for c in expected}
    if "has_audio" in row_local:
        row_local["has_audio"] = 1.0
    if "has_video" in row_local:
        row_local["has_video"] = 1.0
    if "modality_audio_video" in row_local:
        row_local["modality_audio_video"] = 1.0
    if "modality_audio_only" in row_local:
        row_local["modality_audio_only"] = 0.0
    matched_audio = 0
    for k, v in (audio_dict or {}).items():
        if k in row_local:
            row_local[k] = v
            matched_audio += 1
    matched_extra = 0
    for k, v in (extra_dict or {}).items():
        if k in row_local:
            row_local[k] = v
            matched_extra += 1
    debug["matched_audio_cols"] = matched_audio
    debug["matched_extra_cols"] = matched_extra
    debug["expected_cols"] = len(expected)
    return row_local

# Segment-level inference: sample up to 5 one-second windows across the clip
combined_feats = None
if model is not None and expected_cols:
    try:
        total_dur = float(audio_feats.get("duration_sec", 0.0)) if isinstance(audio_feats, dict) else 0.0
        if total_dur >= 7.0:
            offsets = [max(0.0, total_dur * r - 0.5) for r in [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]]
        elif total_dur >= 5.0:
            offsets = [max(0.0, total_dur * r - 0.5) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
        elif total_dur > 1.5:
            offsets = [max(0.0, total_dur * 0.5 - 0.5)]
        else:
            offsets = [0.0]
        seg_rows = []
        for off in offsets:
            try:
                seg = extract_audio_features(temp_path, offset_sec=off, duration_sec=1.0)
                seg_rows.append(_row_from_audio_feats(seg, expected_cols, extra_dict=vstats))
            except Exception:
                continue
        if seg_rows:
            combined_feats = pd.DataFrame(seg_rows, columns=expected_cols)
            debug["audio_segments"] = len(seg_rows)
    except Exception:
        pass

if combined_feats is None:
    if model is not None and expected_cols:
        combined_feats = pd.DataFrame([_row_from_audio_feats(audio_feats, expected_cols, extra_dict=vstats)], columns=expected_cols)
    else:
        combined_feats = pd.DataFrame([{**audio_feats}])




    # === Prediction
if model is not None:
    # Guard: if the constructed row barely overlaps expected columns, avoid misleading outputs
    try:
        overlap = int((combined_feats.iloc[0] != 0).sum())
        if expected_cols and overlap < max(1, int(0.2 * len(expected_cols))):
            st.error("Model features do not match runtime features (very low overlap). Please rebuild the model with fused features.")
            with st.expander("🔧 Debug details"):
                st.write({"matched_nonzero_cols": overlap, "expected_cols": len(expected_cols)})
            st.stop()
    except Exception:
        pass
    st.info("🎯 Predicting emotion...")
    proba_all = model.predict_proba(combined_feats)
    # Aggregate across segments if multiple rows
    if isinstance(proba_all, list):
        proba_all = np.asarray(proba_all)
    if getattr(proba_all, "ndim", 1) == 2 and proba_all.shape[0] > 1:
        proba = proba_all.mean(axis=0)
    else:
        proba = proba_all[0] if hasattr(proba_all, "__getitem__") else np.asarray(proba_all)
    # Majority vote across segments with mean confidence for the winning class
    pred_ids = np.argmax(proba_all, axis=1) if getattr(proba_all, "ndim", 1) == 2 else [int(np.argmax(proba))]
    if isinstance(pred_ids, list) or getattr(pred_ids, "ndim", 1) == 1:
        vals, counts = np.unique(np.asarray(pred_ids), return_counts=True)
        majority_id = int(vals[np.argmax(counts)])
        # Require at least 2 votes to accept
        votes = int(counts[np.argmax(counts)])
        emotion_id = majority_id
        confidence = (
            float(np.mean(proba_all[:, majority_id]))
            if getattr(proba_all, "ndim", 1) == 2 and proba_all.shape[0] > 1
            else float(proba[majority_id])
        )
        if votes < 2 and getattr(proba_all, "ndim", 1) == 2:
            # Keep the computed confidence but flag low agreement
            st.info("Low agreement across segments (votes < 2). Confidence may be unreliable.")
            debug["segment_votes"] = votes
    else:
        emotion_id = int(np.argmax(proba))
        confidence = float(proba[emotion_id])
    if label_classes:
        # Use trained label classes if available
        try:
            emotion = str(label_classes[emotion_id])
        except Exception:
            emotion = label_map.get(emotion_id, "unknown")
    else:
        emotion = label_map.get(emotion_id, "unknown")

    # Top-3 predictions for context
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3 = [(
        str(label_classes[i]) if label_classes else label_map.get(i, str(i)),
        float(proba[i])
    ) for i in top3_idx]
    st.write({"top3": [{"label": t[0], "p": round(t[1], 4)} for t in top3]})

    # Uncertainty gating
    if confidence < class_thresholds.get(emotion_id, 0.4):
        st.warning("Model uncertainty is high (confidence < 40%). Results may be unreliable.")
else:
    emotion = "unknown"
    confidence = 0.0

st.success(f"🧠 Predicted Emotion: **{emotion}**")
st.write(f"🔢 Confidence: `{confidence:.2%}`")

# === Role Matching with class-specific thresholds
st.info("🎭 Matching to character profile...")
thr = max(class_thresholds.get(emotion_id, 0.4), threshold_min) if model is not None else 0.4
matched_role = match_role(emotion) if confidence >= thr else "needs clearer input"
st.success(f"🎬 Recommended Role: **{matched_role}**")

# === SHAP Explanation (per-instance)
try:
    import shap
    if model is not None and shap_background is not None and expected_cols:
        st.subheader("📈 Feature Contribution (SHAP)")
        # Apply same scaling as pipeline
        X_scaled = model.named_steps["scaler"].transform(combined_feats)
        # Always use model-level KernelExplainer for robustness across calibrated wrappers
        clf = model.named_steps["clf"]
        predict_fn = lambda x: clf.predict_proba(x)
        explainer = shap.KernelExplainer(predict_fn, shap_background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)
        # If multiclass, pick predicted class explanation
        if isinstance(shap_values, list):
            shap_values_for_class = shap_values[int(emotion_id)]
        elif getattr(shap_values, "ndim", 1) == 3:
            shap_values_for_class = shap_values[0, :, int(emotion_id)]
        else:
            shap_values_for_class = shap_values
        shap.force_plot = getattr(shap, "force_plot", None)
        # Show top features as a small dataframe for reliability
        vals = np.abs(shap_values_for_class[0] if getattr(shap_values_for_class, "ndim", 1) > 1 else shap_values_for_class)
        order = np.argsort(vals)[-10:][::-1]
        row_vals = shap_values_for_class[0] if getattr(shap_values_for_class, "ndim", 1) > 1 else shap_values_for_class
        top_feats = [(expected_cols[i], float(row_vals[i])) for i in order]
        st.json({"top_shap": [{"feature": n, "shap": round(v, 5)} for n, v in top_feats]})
        try:
            df_shap = pd.DataFrame(top_feats, columns=["feature", "shap"]).iloc[::-1]
            # Matplotlib horizontal bar chart for reliable rendering with signed values
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(6, 3 + 0.25 * len(df_shap)))
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df_shap["shap"].values]
            ax.barh(df_shap["feature"], df_shap["shap"], color=colors)
            ax.axvline(0, color="#999", linewidth=1)
            ax.set_xlabel("SHAP value (impact)")
            ax.set_ylabel("")
            _plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            pass
    elif model is not None and shap_background is None:
        st.info("SHAP background not available. Retrain the model once to enable per-instance explanations.")
except Exception as e:
    # Fallback: show global importances for tree models if SHAP isn't available
    try:
        st.info(f"SHAP not available: {e}. Showing global feature importances instead.")
        clf = model.named_steps["clf"]
        
        # Handle VotingClassifier
        if hasattr(clf, "estimators_"):
            # VotingClassifier - get feature importances from individual estimators
            all_importances = []
            for i, estimator in enumerate(clf.estimators_):
                # If estimator is a pipeline, get the final step
                if hasattr(estimator, 'steps'):
                    final_step = estimator.steps[-1][1]
                else:
                    final_step = estimator
                
                if hasattr(final_step, "feature_importances_"):
                    all_importances.append(final_step.feature_importances_)
            
            if all_importances:
                # Handle different feature dimensions by showing them separately
                if len(set(len(imp) for imp in all_importances)) == 1:
                    # All estimators have same number of features - average them
                    fi = np.mean(all_importances, axis=0)
                    order = np.argsort(fi)[-10:][::-1]
                    top_feats = [(expected_cols[i], float(fi[i])) for i in order]
                    df_shap = pd.DataFrame(top_feats, columns=["feature", "importance"]).iloc[::-1]
                    import matplotlib.pyplot as _plt
                    fig, ax = _plt.subplots(figsize=(6, 3 + 0.25 * len(df_shap)))
                    ax.barh(df_shap["feature"], df_shap["importance"], color="#1f77b4")
                    ax.set_xlabel("Global importance (averaged across estimators)")
                    ax.set_ylabel("")
                    _plt.tight_layout()
                    st.pyplot(fig)
                else:
                    # Different feature dimensions - show the largest one
                    largest_imp = max(all_importances, key=len)
                    order = np.argsort(largest_imp)[-10:][::-1]
                    top_feats = [(expected_cols[i], float(largest_imp[i])) for i in order]
                    df_shap = pd.DataFrame(top_feats, columns=["feature", "importance"]).iloc[::-1]
                    import matplotlib.pyplot as _plt
                    fig, ax = _plt.subplots(figsize=(6, 3 + 0.25 * len(df_shap)))
                    ax.barh(df_shap["feature"], df_shap["importance"], color="#1f77b4")
                    ax.set_xlabel("Global importance (from largest estimator)")
                    ax.set_ylabel("")
                    _plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Cannot render feature importances for VotingClassifier - no tree-based estimators found.")
        else:
            # Regular classifier
            estimator = clf.base_estimator if hasattr(clf, "base_estimator") else clf
            if hasattr(estimator, "feature_importances_"):
                fi = np.asarray(estimator.feature_importances_)
                order = np.argsort(fi)[-10:][::-1]
                top_feats = [(expected_cols[i], float(fi[i])) for i in order]
                df_shap = pd.DataFrame(top_feats, columns=["feature", "importance"]).iloc[::-1]
                import matplotlib.pyplot as _plt
                fig, ax = _plt.subplots(figsize=(6, 3 + 0.25 * len(df_shap)))
                ax.barh(df_shap["feature"], df_shap["importance"], color="#1f77b4")
                ax.set_xlabel("Global importance")
                ax.set_ylabel("")
                _plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Cannot render SHAP or global importances for this model.")
    except Exception:
        st.info("SHAP visualization unavailable.")

    # === Save Prediction to CSV Log
    log_entry = {
        "filename": uploaded_file.name,
        "predicted_emotion": emotion,
        "confidence": confidence,
        "matched_role": matched_role,
        "timestamp": datetime.now().isoformat()
    }
    log_df = pd.DataFrame([log_entry])
    log_path = Path("outputs/prediction_log.csv")
    if log_path.exists():
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
    st.success("📝 Prediction saved to `outputs/prediction_log.csv`")

    #  Debug display
    with st.expander("🔬 Show extracted features"):
        st.write("🎧 Audio Features", pd.DataFrame([audio_feats]))
    st.write("🎥 Video Frames", debug.get("num_video_frames"))

with st.expander("🔧 Debug details"):
    st.write(debug)

def _save_upload(upload):
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        return Path(tmp.name)


