"""
src/train.py
LoanNet — Training script with callbacks, evaluation, and model saving.
Python 3.11 | TensorFlow 2.15
"""

import os, time, argparse
import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)

# ── Local imports ─────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_pipeline import run_pipeline
from src.model import build_loannet


def train(
    csv_path:   str   = "data/loan_data.csv",
    epochs:     int   = 200,
    batch_size: int   = 256,
    dropout1:   float = 0.4,
    dropout2:   float = 0.3,
    lr:         float = 1e-3,
    patience:   int   = 14,
    val_pct:    float = 0.15,
    model_dir:  str   = "models",
    verbose:    int   = 1,
):
    os.makedirs(model_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────
    print("\n── Data Pipeline ───────────────────────────────")
    X_tr, X_va, X_te, y_tr, y_va, y_te, features = run_pipeline(csv_path, val_pct)
    print(f"Features ({len(features)}): {features}")

    # ── Model ────────────────────────────────────────────────
    print("\n── Model Architecture ──────────────────────────")
    model = build_loannet(input_dim=len(features), dropout1=dropout1,
                          dropout2=dropout2, lr=lr)
    model.summary()

    # ── Callbacks ────────────────────────────────────────────
    model_path = os.path.join(model_dir, "loannet_best.h5")
    callbacks  = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
            histogram_freq=0
        ),
        keras.callbacks.CSVLogger(
            os.path.join(model_dir, "training_log.csv"),
            append=False
        ),
    ]

    # ── Train ────────────────────────────────────────────────
    print("\n── Training ────────────────────────────────────")
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    elapsed = time.time() - t0
    best_ep  = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nTraining complete: {elapsed:.1f}s  |  Best epoch: {best_ep}")

    # ── Evaluate ─────────────────────────────────────────────
    print("\n── Test Set Evaluation ─────────────────────────")
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc    = accuracy_score(y_te, y_pred) * 100
    auc    = roc_auc_score(y_te, y_prob)
    f1     = f1_score(y_te, y_pred, zero_division=0) * 100
    cm     = confusion_matrix(y_te, y_pred)

    print(f"  Accuracy : {acc:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1 Score : {f1:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred,
          target_names=["Rejected", "Approved"], zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    print(f"\nModel saved → {model_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoanNet")
    parser.add_argument("--data",     default="data/loan_data.csv")
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--batch",    type=int,   default=256)
    parser.add_argument("--dropout1", type=float, default=0.4)
    parser.add_argument("--dropout2", type=float, default=0.3)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=14)
    args = parser.parse_args()

    train(
        csv_path=args.data, epochs=args.epochs,
        batch_size=args.batch, dropout1=args.dropout1,
        dropout2=args.dropout2, lr=args.lr, patience=args.patience
    )
