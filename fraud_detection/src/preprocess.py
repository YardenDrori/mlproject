import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def load_and_split(data_path, scaler_path):
    """
    Load creditcard.csv, scale Time/Amount, split normals 80/20.

    Returns:
        X_ae_train  - 80% of normals (autoencoder training)
        X_eval_pool - 20% normals + ALL fraud (SVM eval pool)
        y_eval_pool - labels for eval pool
        scaler      - fitted StandardScaler
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it at fraud_detection/data/creditcard.csv"
        )

    df = pd.read_csv(data_path)

    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Scale Time (col 0) and Amount (col 29); V1-V28 are already PCA-scaled
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values

    normal_idx = np.where(y == 0)[0]
    fraud_idx = np.where(y == 1)[0]

    # Shuffle normals with fixed seed
    rng = np.random.default_rng(42)
    shuffled_normals = rng.permutation(normal_idx)
    split_point = int(len(shuffled_normals) * 0.8)

    ae_train_idx = shuffled_normals[:split_point]
    eval_normal_idx = shuffled_normals[split_point:]

    X_ae_train = X[ae_train_idx]

    # Eval pool: 20% normals + ALL fraud
    eval_idx = np.concatenate([eval_normal_idx, fraud_idx])
    eval_idx = rng.permutation(eval_idx)

    X_eval_pool = X[eval_idx]
    y_eval_pool = y[eval_idx]

    total = len(y)
    n_fraud = len(fraud_idx)
    n_normal = len(normal_idx)
    fraud_rate = n_fraud / total * 100

    print(f"  Dataset: {total:,} rows | {n_normal:,} normal | {n_fraud:,} fraud ({fraud_rate:.3f}%)")
    print(f"  AE train:  {len(X_ae_train):,} rows (normals only)")
    print(f"  Eval pool: {len(X_eval_pool):,} rows ({len(eval_normal_idx):,} normal + {n_fraud:,} fraud)")
    print(f"  Scaler saved to: {scaler_path}")

    return X_ae_train, X_eval_pool, y_eval_pool, scaler
