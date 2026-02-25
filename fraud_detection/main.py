import os
import sys
import numpy as np

# Allow running as: python fraud_detection/main.py  (from repo root)
# or:               python main.py                  (from fraud_detection/)
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_and_split
from src.autoencoder import train_autoencoder, compute_reconstruction_errors
from src.svm_classifier import split_data, train_xgb, get_predictions
# from src.svm_classifier import train_xgb_tuned, train_xgb_combined, train_baseline_xgb
# from src.evaluate import run_evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
AE_PATH      = os.path.join(MODELS_DIR, "autoencoder.keras")
XGB_PATH     = os.path.join(MODELS_DIR, "xgb_model.pkl")
XGB_TUNED_PATH     = os.path.join(MODELS_DIR, "xgb_model_tuned.pkl")
XGB_COMBINED_PATH  = os.path.join(MODELS_DIR, "xgb_model_combined.pkl")
BASELINE_PATH      = os.path.join(MODELS_DIR, "xgb_baseline.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Stage 1 — Load & preprocess                                                  #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Stage 1: Load and preprocess data")
print("=" * 60)
X_ae_train, X_eval_pool, y_eval_pool, scaler = load_and_split(DATA_PATH, SCALER_PATH)

# --------------------------------------------------------------------------- #
#  Stage 2 — Train autoencoder (normal transactions only)                       #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Stage 2: Train autoencoder")
print("=" * 60)
ae_model, history = train_autoencoder(X_ae_train, AE_PATH)

# --------------------------------------------------------------------------- #
#  Stage 3 — Compute reconstruction errors on eval pool                         #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Stage 3: Compute reconstruction errors")
print("=" * 60)
errors_eval = compute_reconstruction_errors(ae_model, X_eval_pool)
print(f"  Errors computed for {len(errors_eval):,} samples")
print(f"  Normal  mean error: {errors_eval[y_eval_pool == 0].mean():.6f}")
print(f"  Fraud   mean error: {errors_eval[y_eval_pool == 1].mean():.6f}")

# --------------------------------------------------------------------------- #
#  Stage 4 — Train XGBoost classifiers                                          #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Stage 4: Train XGBoost classifiers")
print("=" * 60)
splits = split_data(errors_eval, X_eval_pool, y_eval_pool)

print(f"\n  Split: train={len(splits['y_train']):,}  test={len(splits['y_test']):,}")
print(f"  Fraud in train: {splits['y_train'].sum()} | test: {splits['y_test'].sum()}")

print("\n  --- AE+XGBoost ---")
ae_xgb = train_xgb(splits["errors_train"], splits["y_train"], XGB_PATH)

# print("\n  --- AE+XGBoost (new: SMOTE 0.3 + GridSearchCV) ---")
# ae_xgb_tuned = train_xgb_tuned(splits["errors_train"], splits["y_train"], XGB_TUNED_PATH)

# print("\n  --- AE+XGBoost (combined: raw + error features) ---")
# ae_xgb_combined = train_xgb_combined(splits["X_train"], splits["errors_train"], splits["y_train"], XGB_COMBINED_PATH)

# print("\n  --- Baseline XGBoost (raw features) ---")
# baseline_xgb = train_baseline_xgb(splits["X_train"], splits["y_train"], BASELINE_PATH)

# --------------------------------------------------------------------------- #
#  Stage 5 — Evaluate                                                           #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Stage 5: Evaluate and generate plots")
print("=" * 60)
y_pred, y_scores = get_predictions(ae_xgb, splits["errors_test"], is_error_based=True)
y_test = splits["y_test"]

print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  AUC:       {roc_auc_score(y_test, y_scores):.4f}")

# combined_test = np.concatenate([splits["X_test"], splits["errors_test"]], axis=1)
# y_pred_ae_new,  y_scores_ae_new  = get_predictions(ae_xgb_tuned,    splits["errors_test"], is_error_based=True)
# y_pred_ae_comb, y_scores_ae_comb = get_predictions(ae_xgb_combined, combined_test,         is_error_based=False)
# y_pred_bl,      y_scores_bl      = get_predictions(baseline_xgb,    splits["X_test"],      is_error_based=False)
# run_evaluation(...)

print("\nDone. Results saved to fraud_detection/results/")
