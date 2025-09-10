import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    log_loss,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    top_k_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import re
 
# === Utilities ===
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
DATA_PATH = Path("features/fused_features.csv")
MODEL_PATH = Path("models/classifier.pkl")
METRICS_PATH = Path("outputs/eval_metrics.json")
CONFUSION_MATRIX_PATH = Path("outputs/confusion_matrix.png")
CONFUSION_MATRIX_NORM_PATH = Path("outputs/confusion_matrix_normalized.png")
ROC_CURVE_PATH = Path("outputs/roc_curves.png")
PR_CURVE_PATH = Path("outputs/pr_curves.png")
CONFIDENCE_OUTPUT_PATH = Path("outputs/predictions_with_confidence.csv")

def train_and_save_model():
    # Ensure fused features exist; try to auto-generate if missing
    if not DATA_PATH.exists():
        try:
            from src.fuse_features import main as build_fused_features
            print(f"[WARN] {DATA_PATH} not found. Attempting to generate fused features...")
            build_fused_features()
        except Exception as e:
            raise FileNotFoundError(
                f"Required dataset '{DATA_PATH}' not found and auto-generation failed. "
                f"Please verify your dataset CSVs under data/processed/ and rerun fuse_features. Error: {e}"
            )

    # === Load and preprocess ===
    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df, columns=["modality"])  # one-hot encode modality
    video_ids = df["video_id"]  # store before dropping
    # Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(df["label"].astype(str))
    if len(le.classes_) < 2:
        raise ValueError("Training data contains <2 classes. Rebuild fused features with balanced sampling.")
    # Features (drop target-related columns)
    drop_cols = [c for c in ["label", "video_id", "emotion_id", "filepath"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # === Split data ===
    # Prefer GroupKFold when video_ids represent speakers/utterances
    try:
        gkf = GroupKFold(n_splits=5)
        # Use the first split
        train_idx, test_idx = next(gkf.split(X, y_encoded, groups=video_ids))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
        vid_train, vid_test = video_ids.iloc[train_idx], video_ids.iloc[test_idx]
    except Exception:
        X_train, X_test, y_train, y_test, vid_train, vid_test = train_test_split(
            X, y_encoded, video_ids, test_size=0.2, random_state=42, stratify=y_encoded
        )

    # === Determine modality columns ===
    audio_cols = [c for c in X.columns if re.search(r"(audio|mfcc|chroma|spectral|zcr|tempo|mel|librosa)", c, re.I)]
    video_cols = [c for c in X.columns if re.search(r"(video|frame|visual|embedding)", c, re.I)]
    # Fallbacks if regex fails
    if not audio_cols:
        audio_cols = [c for c in X.columns if "audio" in c.lower()]
    if not video_cols:
        video_cols = [c for c in X.columns if "video" in c.lower() or "frame" in c.lower()]

    # === Train model (pipeline) with optional CV model selection ===
    tune_cv = bool(int(os.environ.get("TUNE_CV", "1")))
    kbest_k = min(80, X.shape[1])

    if tune_cv:
        candidate_specs = []
        # Fast mode: two strong baselines, small configs
        rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1, max_depth=None)
        hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300, random_state=42)

        for name, clf in [
            ("RF", rf),
            ("HGB", hgb),
        ]:
            # With feature selection
            pipe_sel = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("var", VarianceThreshold(threshold=0.0)),
                ("selector", SelectKBest(score_func=f_classif, k=kbest_k)),
                ("clf", clf),
            ])
            candidate_specs.append((f"{name}+KBest", pipe_sel))

            # Without feature selection
            pipe_nosel = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("var", VarianceThreshold(threshold=0.0)),
                ("clf", clf),
            ])
            candidate_specs.append((f"{name}", pipe_nosel))

        cv = GroupKFold(n_splits=3)
        scores = []
        for cname, cpipeline in candidate_specs:
            try:
                s = cross_val_score(cpipeline, X_train, y_train, cv=cv, groups=vid_train, scoring="accuracy")
                scores.append((cname, float(np.mean(s)), float(np.std(s))))
            except Exception as e:
                scores.append((cname, -1.0, 0.0))

        # Pick best by mean accuracy
        scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
        best_name = scores_sorted[0][0]
        print("[INFO] CV candidates (mean±std accuracy):")
        for n, m, sd in scores_sorted:
            print(f"  - {n}: {m:.3f} ± {sd:.3f}")
        # Recreate the best pipeline and add calibration for final fit
        if best_name.startswith("RF"):
            chosen = rf
        else:
            chosen = hgb
        calibrated = CalibratedClassifierCV(estimator=chosen, cv=3, method="sigmoid")
        if "+KBest" in best_name:
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("selector", SelectKBest(score_func=f_classif, k=kbest_k)),
                ("clf", calibrated),
            ])
        else:
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", calibrated),
            ])
        model_base = f"{type(chosen).__name__} ({'KBest' if '+KBest' in best_name else 'NoKBest'})"
    else:
        # Flags
        use_gb = bool(int(os.environ.get("TRAIN_USE_GB", "0")))
        enable_fusion = bool(int(os.environ.get("LATE_FUSION", "1")))
        fusion_mode = os.environ.get("FUSION_MODE", "voting").strip().lower()
        fusion_weights_env = os.environ.get("FUSION_WEIGHTS", "0.8,0.2")
        fusion_tune = bool(int(os.environ.get("FUSION_TUNE", "0")))
        try:
            fusion_weights = [float(x) for x in fusion_weights_env.split(",")]
            if len(fusion_weights) != 2:
                fusion_weights = [0.8, 0.2]
        except Exception:
            fusion_weights = [0.8, 0.2]
        enable_oversample = bool(int(os.environ.get("OVERSAMPLE", "1")))

        # Base estimator factory
        def make_base_estimator():
            if use_gb:
                return GradientBoostingClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                )
            else:
                return RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )

        # ColumnSelector defined at module scope for pickling

        if enable_fusion and audio_cols and video_cols:
            # Build audio branch
            audio_branch = Pipeline([
                ("select", ColumnSelector(audio_cols)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("var", VarianceThreshold(threshold=0.0)),
                ("selector", SelectKBest(score_func=f_classif, k=min(kbest_k, max(1, len(audio_cols) - 1)))),
                ("clf", make_base_estimator()),
            ])
            # Build video branch
            video_branch = Pipeline([
                ("select", ColumnSelector(video_cols)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("var", VarianceThreshold(threshold=0.0)),
                ("selector", SelectKBest(score_func=f_classif, k=min(kbest_k, max(1, len(video_cols) - 1)))),
                ("clf", make_base_estimator()),
            ])
            # Optional quick fusion tuner
            if fusion_tune:
                cv = GroupKFold(n_splits=3)
                audio_models = [
                    ("RF", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)),
                    ("HGB", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300, random_state=42)),
                ]
                video_models = [
                    ("RF", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)),
                    ("HGB", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300, random_state=42)),
                ]
                weight_sets = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]

                def build_branch(columns, est):
                    return Pipeline([
                        ("select", ColumnSelector(columns)),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=True)),
                        ("var", VarianceThreshold(threshold=0.0)),
                        ("clf", est),
                    ])

                candidates = []
                for aname, aest in audio_models:
                    for vname, vest in video_models:
                        for w in weight_sets:
                            est = VotingClassifier(
                                estimators=[("audio", build_branch(audio_cols, aest)), ("video", build_branch(video_cols, vest))],
                                voting="soft",
                                weights=list(w),
                            )
                            if enable_oversample:
                                model = ImbPipeline([( "sampler", RandomOverSampler(random_state=42)), ("clf", est)])
                            else:
                                model = ImbPipeline([( "clf", est)])
                            try:
                                s = cross_val_score(model, X_train, y_train, cv=cv, groups=vid_train, scoring="accuracy")
                                candidates.append(((aname, vname, w), float(s.mean())))
                            except Exception:
                                candidates.append(((aname, vname, w), -1.0))
                candidates.sort(key=lambda t: t[1], reverse=True)
                best = candidates[0][0]
                audio_choice = audio_models[0][1] if best[0] == "RF" else audio_models[1][1]
                video_choice = video_models[0][1] if best[1] == "RF" else video_models[1][1]
                fusion_weights = list(best[2])
                # Rebuild branches with chosen models
                audio_branch = build_branch(audio_cols, audio_choice)
                video_branch = build_branch(video_cols, video_choice)
                print(f"[INFO] Fusion tuner best -> audio={best[0]}, video={best[1]}, weights={fusion_weights}, acc={candidates[0][1]:.3f}")

            if fusion_mode == "stacking":
                fusion_est = StackingClassifier(
                    estimators=[("audio", audio_branch), ("video", video_branch)],
                    final_estimator=LogisticRegression(max_iter=1000),
                    stack_method="predict_proba",
                    passthrough=False,
                )
                model_base = "LateFusion(Stacking)"
            else:
                fusion_est = VotingClassifier(
                    estimators=[("audio", audio_branch), ("video", video_branch)],
                    voting="soft",
                    weights=fusion_weights,
                )
                model_base = f"LateFusion(Voting weights={fusion_weights})"

            if enable_oversample:
                pipeline = ImbPipeline([
                    ("sampler", RandomOverSampler(random_state=42)),
                    ("clf", fusion_est),
                ])
            else:
                pipeline = ImbPipeline([
                    ("clf", fusion_est),
                ])
        else:
            # Single-branch pipeline on all features
            base_clf = make_base_estimator()
            calibrated = CalibratedClassifierCV(estimator=base_clf, cv=3, method="sigmoid")
            core_steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("selector", SelectKBest(score_func=f_classif, k=kbest_k)),
                ("clf", calibrated),
            ]
            if enable_oversample:
                pipeline = ImbPipeline([
                    ("sampler", RandomOverSampler(random_state=42)),
                    *core_steps,
                ])
            else:
                pipeline = ImbPipeline(core_steps)
            model_base = type(base_clf).__name__

    pipeline.fit(X_train, y_train)

    # === Save model ===
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # === Compute per-class probability thresholds on validation set ===
    # Use Youden's J (tpr - fpr) to choose thresholds. Fallback to 0.4
    thresholds = {}
    try:
        min_thr = float(os.environ.get("THR_MIN", "0.35"))
    except Exception:
        min_thr = 0.35
    try:
        # probabilities on X_test already computed below, recompute here after fit
        proba_val = pipeline.predict_proba(X_test)
        num_classes = proba_val.shape[1]
        for c in range(num_classes):
            y_true_c = (y_test.values == c).astype(int)
            fpr, tpr, thr = roc_curve(y_true_c, proba_val[:, c])
            if thr is not None and len(thr) > 0:
                j = tpr - fpr
                best_idx = int(j.argmax())
                thr_c = float(thr[best_idx])
                thresholds[c] = max(min_thr, min(0.8, thr_c))
            else:
                thresholds[c] = max(min_thr, 0.4)
    except Exception:
        for c in range(len(le.classes_)):
            thresholds[c] = max(min_thr, 0.4)

    # Prepare SHAP background on scaled training features for fast per-instance explainability
    try:
        X_train_scaled = pipeline[:-1].transform(X_train)
        rng = np.random.RandomState(42)
        take = min(100, X_train_scaled.shape[0])
        idx = rng.choice(X_train_scaled.shape[0], size=take, replace=False)
        shap_background = X_train_scaled[idx]
    except Exception:
        shap_background = None

    bundle = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "label_classes": list(le.classes_),
        "thresholds": thresholds,
        "threshold_min": min_thr,
        "shap_background": shap_background,
        "model_base": model_base,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    # === Predictions + Confidence Scores ===
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)
    confidence_scores = y_proba.max(axis=1)

    # === Compute extended metrics ===
    metrics = {}
    # Basic report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report
    metrics["accuracy"] = float(report.get("accuracy", np.mean(y_pred == y_test)))
    # Accuracies
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    # Top-k (k=2,3 if possible)
    try:
        metrics["top_2_accuracy"] = float(top_k_accuracy_score(y_test, y_proba, k=min(2, y_proba.shape[1])))
    except Exception:
        metrics["top_2_accuracy"] = None
    try:
        metrics["top_3_accuracy"] = float(top_k_accuracy_score(y_test, y_proba, k=min(3, y_proba.shape[1])))
    except Exception:
        metrics["top_3_accuracy"] = None
    # Precision/Recall/F1 summaries
    for avg in ["micro", "macro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)
    # Prob-based metrics
    try:
        metrics["log_loss"] = float(log_loss(y_test, y_proba))
    except Exception:
        metrics["log_loss"] = None
    # ROC-AUC (OvR macro and per-class)
    try:
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        per_class_roc = {}
        num_classes = y_proba.shape[1]
        for c in range(num_classes):
            y_true_c = (y_test.values == c).astype(int)
            try:
                per_class_roc[str(c)] = float(roc_auc_score(y_true_c, y_proba[:, c]))
            except Exception:
                per_class_roc[str(c)] = None
        metrics["roc_auc_per_class"] = per_class_roc
    except Exception:
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_ovr_weighted"] = None
        metrics["roc_auc_per_class"] = None
    # PR AUC macro (average_precision)
    try:
        metrics["pr_auc_macro"] = float(average_precision_score(pd.get_dummies(y_test).values, y_proba, average="macro"))
        metrics["pr_auc_weighted"] = float(average_precision_score(pd.get_dummies(y_test).values, y_proba, average="weighted"))
    except Exception:
        metrics["pr_auc_macro"] = None
        metrics["pr_auc_weighted"] = None
    # Agreement metrics
    try:
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_test, y_pred))
    except Exception:
        metrics["cohen_kappa"] = None
    try:
        metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_test, y_pred))
    except Exception:
        metrics["matthews_corrcoef"] = None

    # Save metrics JSON
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Evaluation metrics saved to {METRICS_PATH}")

    # === Save confusion matrix ===
    cm = confusion_matrix(y_test, y_pred)
    labels = list(le.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"[INFO] Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # Normalized confusion matrix
    try:
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Normalized)")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_NORM_PATH)
        print(f"[INFO] Normalized confusion matrix saved to {CONFUSION_MATRIX_NORM_PATH}")
    except Exception:
        pass

    # ROC curves (one-vs-rest)
    try:
        num_classes = y_proba.shape[1]
        plt.figure(figsize=(8, 6))
        for c in range(num_classes):
            y_true_c = (y_test.values == c).astype(int)
            fpr, tpr, _ = roc_curve(y_true_c, y_proba[:, c])
            plt.plot(fpr, tpr, label=f"{labels[c]}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (OvR)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_CURVE_PATH)
        print(f"[INFO] ROC curves saved to {ROC_CURVE_PATH}")
    except Exception:
        pass

    # Precision-Recall curves (one-vs-rest)
    try:
        from sklearn.metrics import precision_recall_curve
        num_classes = y_proba.shape[1]
        plt.figure(figsize=(8, 6))
        for c in range(num_classes):
            y_true_c = (y_test.values == c).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_c, y_proba[:, c])
            plt.plot(recall, precision, label=f"{labels[c]}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (OvR)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(PR_CURVE_PATH)
        print(f"[INFO] PR curves saved to {PR_CURVE_PATH}")
    except Exception:
        pass

    # === Save prediction output with confidence scores ===
    predicted_labels = [labels[pred] for pred in y_pred]
    true_labels = [labels[true] for true in y_test]

    results_df = pd.DataFrame({
        "video_id": vid_test.values,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "confidence_score": confidence_scores
    })
    CONFIDENCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CONFIDENCE_OUTPUT_PATH, index=False)
    print(f"[INFO] Prediction results with confidence saved to {CONFIDENCE_OUTPUT_PATH}")


def load_trained_model(path: Path = None):
    """Load model bundle: pipeline, feature_names, label_classes."""
    model_path = path or MODEL_PATH
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    return joblib.load(model_path)


if __name__ == "__main__":
    train_and_save_model()
