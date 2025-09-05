# AI Casting Assistant

A multimodal (audio + video) emotion analysis prototype that helps match audition clips to role descriptors. The project includes feature fusion, leakage‑free training with late fusion and oversampling, calibrated predictions with abstention, explainability, and a Streamlit UI.

## 1. Quick start

```bash
# Create and activate a virtual environment (macOS)
/usr/bin/python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# All dependencies including imbalanced-learn are in requirements.txt
```

## 2. Data prep and feature fusion

Place metadata CSVs under `data/processed/`:
- `CREMA-D_metadata.csv`, `EMOVO_metadata.csv`, `RAVDESS_metadata.csv`
  - Required columns: `filepath`, `label`, `modality` (audio_only or audio_video)

Generate/refresh fused features:
```bash
# Defaults: balanced sampling off, no global cap
python -m src.fuse_features

# Useful environment flags
FUSE_PER_CLASS=300 \   # cap per label before fusion (balanced sampling)
FUSE_MAX_ROWS=1500 \   # global cap for faster development
FUSE_SEGMENT_SEC=2.0 \ # compute audio on multiple 2s offsets and aggregate
python -m src.fuse_features
```
Artifacts:
- `features/fused_features.csv`

## 3. Training (multimodal, leakage‑free)

Late fusion and class‑imbalance handling are configurable via environment variables.

```bash
# Multimodal late fusion (soft voting), oversampling in training folds, RandomForest base
LATE_FUSION=1 \
FUSION_MODE=voting \
FUSION_WEIGHTS=0.8,0.2 \
OVERSAMPLE=1 \
TRAIN_USE_GB=0 \
TUNE_CV=0 \
python -m src.train_classifier
```

Flags:
- `LATE_FUSION={0|1}`: enable late fusion (audio and video branches)
- `FUSION_MODE={voting|stacking}`: soft voting or stacking meta‑learner (LR)
- `FUSION_WEIGHTS=a,b`: voting weights for audio,video (e.g. `0.9,0.1`)
- `OVERSAMPLE={0|1}`: leakage‑free `RandomOverSampler` inside the train pipeline
- `TRAIN_USE_GB={0|1}`: use GradientBoosting (1) or RandomForest (0)
- `TUNE_CV={0|1}` / `FUSION_TUNE={0|1}`: lightweight CV model/fusion search (optional)

Outputs:
- `models/classifier.pkl` (pipeline bundle + thresholds)
- `outputs/eval_metrics.json`
- `outputs/confusion_matrix.png`, `outputs/confusion_matrix_normalized.png`
- `outputs/predictions_with_confidence.csv`

## 4. Explainability

The Streamlit app renders SHAP explanations when available. The training step stores a small background set to accelerate SHAP.

Fallback: global feature importances for tree models when SHAP is unavailable.

## 5. Streamlit app

```bash
streamlit run dashboard.py
```
Features:
- Upload → preview → feature extraction → prediction → explanation → role suggestion
- Majority voting on multi‑segment clips; confidence display and abstention
- Logs to `outputs/prediction_log.csv`

## 6. SUS (usability) workbook

A spreadsheet template with formulas is provided:
- `outputs/sus_calculation.xlsx`
  - Sheet `SUS_Raw`: enter Q1..Q10 (Likert 1–5). Odd items score `Q-1`, even items `5-Q`. SUS per participant = `(OddSum + EvenSum) * 2.5`.
  - Sheet `Summary`: automatic `n`, mean SUS, std.

## 7. Current results (multimodal, expanded dataset)

Latest held‑out evaluation (late fusion voting, oversampling, no leakage):
- Accuracy: 0.490
- Balanced accuracy: 0.502
- Macro F1: 0.486; Weighted F1: 0.479
- Top‑2: 0.690; Top‑3: 0.857
- Log loss: 1.385; κ: 0.387; MCC: 0.391

Artifacts for reporting:
- `outputs/metrics_table_summary.png`
- `outputs/metrics_table_classwise.png`
- `outputs/metrics_table.tex` (\input into LaTeX)

## 8. Known limitations
- Dataset scale and domain shift; class imbalance persists in the wild
- Video branch uses lightweight frame statistics; expression‑specific embeddings not yet integrated
- Multimodal accuracy < 0.70; fusion improves with more/better video features and data

## 9. Roadmap
- Integrate face‑cropped, expression‑specific embeddings for the video branch
- Expand dataset (≥1–2k fused rows) with balanced per‑class, per‑speaker splits
- Small CV sweep over fusion weights and branch models; drop `SelectKBest` as scale grows
- Formal SUS and fairness studies with grouped metrics

## 10. Repro tips
- Always keep oversampling inside the training pipeline (never touch test)
- Use `GroupKFold` by `video_id` to avoid identity leakage
- Version fused features when changing env flags (e.g., `features/fused_features.v2.csv`)
# Casting-assistant-
