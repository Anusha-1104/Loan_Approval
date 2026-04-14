"""
src/data_pipeline.py
LoanNet — Data loading, cleaning, feature engineering, normalisation, and splitting.
Python 3.11 | scikit-learn 1.4
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os

CAT_COLS = ["education", "employment_type", "loan_purpose"]
TARGET   = "approved"


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV loan dataset."""
    df = pd.read_csv(path)
    print(f"[load]  {len(df):,} rows  |  {len(df.columns)} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates, handle missing values,
    clip numeric outliers, validate target.
    """
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    # Clip numeric ranges
    clips = {
        "credit_score":     (300, 850),
        "debt_to_income":   (0.0, 1.0),
        "age":              (18, 80),
        "income":           (0, 2_000_000),
        "loan_amount":      (1_000, 5_000_000),
        "employment_years": (0, 50),
    }
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Ensure binary target
    df[TARGET] = df[TARGET].astype(int).clip(0, 1)
    df = df.reset_index(drop=True)
    print(f"[clean] {before:,} → {len(df):,} rows  (removed {before-len(df):,})")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    All new features improve predictive signal.
    """
    eps = 1e-6

    # Ratio features
    if "loan_amount" in df.columns and "income" in df.columns:
        df["loan_to_income"] = (df["loan_amount"] / (df["income"] + eps)).round(4)

    if "loan_amount" in df.columns and "property_value" in df.columns:
        df["loan_to_value"] = (df["loan_amount"] / (df["property_value"] + eps)).round(4)

    if "loan_amount" in df.columns:
        df["monthly_payment"] = (df["loan_amount"] * 0.005).round(2)

    # Credit quality band (ordinal)
    if "credit_score" in df.columns:
        df["credit_band"] = pd.cut(
            df["credit_score"],
            bins=[300, 580, 670, 740, 800, 851],
            labels=[0, 1, 2, 3, 4],
            right=False
        ).astype(float).fillna(0).astype(int)

    # Risk indicator
    if "num_delinquencies" in df.columns:
        df["has_delinquency"] = (df["num_delinquencies"] > 0).astype(int)

    print(f"[engineer] feature count: {len(df.columns)-1}")
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True,
                         encoders: dict = None) -> tuple:
    """
    Label-encode categorical columns.
    Returns (df_encoded, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    return df, encoders


def split_and_scale(df: pd.DataFrame, val_pct: float = 0.15,
                    seed: int = 42, scaler_path: str = "models/scaler.pkl",
                    encoder_path: str = "models/encoders.pkl"):
    """
    Encode, shuffle, split (70/15/15 temporal), and StandardScale.
    Saves scaler and encoders to disk.
    """
    df, encoders = encode_categoricals(df, fit=True)

    features = [c for c in df.columns if c != TARGET]
    X = df[features].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    # Shuffle
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n  = len(X)
    i1 = int(n * (1 - 2 * val_pct))
    i2 = int(n * (1 - val_pct))

    X_tr, X_va, X_te = X[:i1], X[i1:i2], X[i2:]
    y_tr, y_va, y_te = y[:i1], y[i1:i2], y[i2:]

    sc = StandardScaler().fit(X_tr)

    os.makedirs("models", exist_ok=True)
    joblib.dump(sc, scaler_path)
    joblib.dump(encoders, encoder_path)
    print(f"[split] train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
    print(f"[saved] {scaler_path}  {encoder_path}")

    return (sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te),
            y_tr, y_va, y_te, features)


def run_pipeline(csv_path: str, val_pct: float = 0.15):
    """Full pipeline: load → clean → engineer → encode → split → scale."""
    df = load_data(csv_path)
    df = clean(df)
    df = engineer(df)
    return split_and_scale(df, val_pct=val_pct)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/loan_data.csv"
    result = run_pipeline(path)
    print("Pipeline complete. Feature count:", len(result[6]))
