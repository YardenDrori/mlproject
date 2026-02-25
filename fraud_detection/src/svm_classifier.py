import os
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


def split_data(errors_eval, X_eval_pool, y_eval_pool):
    """
    Stratified 80/20 split of the eval pool.
    Stratification preserves the ~100:1 class imbalance in both splits.

    Returns dict with keys:
        errors_train, errors_test  - per-feature reconstruction errors (n, 30)
        X_train, X_test            - raw 30-feature arrays
        y_train, y_test            - labels
    """
    indices = np.arange(len(y_eval_pool))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_eval_pool,
        random_state=42,
    )

    return {
        "errors_train": errors_eval[train_idx],
        "errors_test":  errors_eval[test_idx],
        "X_train":      X_eval_pool[train_idx],
        "X_test":       X_eval_pool[test_idx],
        "y_train":      y_eval_pool[train_idx],
        "y_test":       y_eval_pool[test_idx],
    }


def _make_xgb(scale_pos_weight):
    return XGBClassifier(
        device="cuda",
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        eval_metric="logloss",
        random_state=42,
    )


def train_xgb(errors_train, y_train, model_path):
    """
    Train XGBoost on per-feature reconstruction errors (30 features).
    scale_pos_weight handles class imbalance (~100:1).
    device='cuda' runs on GPU.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = _make_xgb(scale_pos_weight)
    print(f"  Fitting AE+XGBoost on {len(y_train):,} samples, {errors_train.shape[1]} error features…")
    print(f"  scale_pos_weight = {scale_pos_weight:.1f}")
    clf.fit(errors_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"  AE+XGBoost saved to: {model_path}")
    return clf


def train_xgb_tuned(errors_train, y_train, model_path):
    """
    AE+XGBoost (new): SMOTE(0.3) + GridSearchCV over XGBoost hyperparams.

    Uses imblearn Pipeline so SMOTE is applied only to training folds during
    cross-validation — no leakage into validation folds.

    sampling_strategy=0.3 means fraud is oversampled to 30% of normal count
    (~1:3.3 ratio), which reduces false alarms vs the default 1:1.

    n_jobs=1: GPU models must not run in parallel processes.
    """
    pipeline = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
        ("xgb", XGBClassifier(device="cuda", eval_metric="logloss", random_state=42)),
    ])

    param_grid = {
        "xgb__n_estimators":  [100, 300],
        "xgb__max_depth":     [3, 6],
        "xgb__learning_rate": [0.05, 0.1, 0.3],
        "xgb__subsample":     [0.8, 1.0],
    }

    n_combos = 2 * 2 * 3 * 2
    print(f"  Grid search: {n_combos} combinations × 5 folds = {n_combos * 5} fits")
    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, n_jobs=1, verbose=1)
    grid.fit(errors_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(grid.best_estimator_, model_path)
    print(f"  AE+XGBoost (tuned) saved to: {model_path}")
    return grid.best_estimator_


def train_xgb_combined(X_train, errors_train, y_train, model_path):
    """
    AE+XGBoost (combined): concatenate raw 30 features + 30 error features → 60 features.
    XGBoost sees both what the transaction looks like AND where the autoencoder struggled.
    Identical SMOTE 0.3 + GridSearchCV setup as train_xgb_tuned.
    """
    X_combined = np.concatenate([X_train, errors_train], axis=1)

    pipeline = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
        ("xgb", XGBClassifier(device="cuda", eval_metric="logloss", random_state=42)),
    ])

    param_grid = {
        "xgb__n_estimators":  [100, 300],
        "xgb__max_depth":     [3, 6],
        "xgb__learning_rate": [0.05, 0.1, 0.3],
        "xgb__subsample":     [0.8, 1.0],
    }

    n_combos = 2 * 2 * 3 * 2
    print(f"  Fitting AE+XGBoost (combined) on {len(y_train):,} samples, {X_combined.shape[1]} features…")
    print(f"  Grid search: {n_combos} combinations × 5 folds = {n_combos * 5} fits")
    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, n_jobs=1, verbose=1)
    grid.fit(X_combined, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(grid.best_estimator_, model_path)
    print(f"  AE+XGBoost (combined) saved to: {model_path}")
    return grid.best_estimator_


def train_baseline_xgb(X_train, y_train, model_path):
    """
    Train XGBoost on raw 30 features (baseline comparison).
    Same config as AE+XGBoost for a fair comparison.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = _make_xgb(scale_pos_weight)
    print(f"  Fitting Baseline XGBoost on {len(y_train):,} samples, {X_train.shape[1]} features…")
    print(f"  scale_pos_weight = {scale_pos_weight:.1f}")
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"  Baseline XGBoost saved to: {model_path}")
    return clf


def get_predictions(clf, X_or_errors, is_error_based):
    """
    Get binary predictions and fraud probability scores.

    Args:
        clf            - fitted XGBClassifier
        X_or_errors    - raw features (n, 30) or per-feature errors (n, 30)
        is_error_based - unused; kept for API compatibility

    Returns:
        y_pred   - binary class labels
        y_scores - probability of fraud (class 1)
    """
    y_pred = clf.predict(X_or_errors)
    y_scores = clf.predict_proba(X_or_errors)[:, 1]
    return y_pred, y_scores
