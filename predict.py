"""
src/predict.py
LoanNet — Inference module for new loan applicants.
Python 3.11 | TensorFlow 2.15
"""

import numpy as np
import joblib
from tensorflow import keras

# ── Load artefacts once at module import ─────────────────────
_model    = None
_scaler   = None
_encoders = None
_features = None   # set after first training run

DECISIONS = [
    (0.70, 1.01, "APPROVED",
     "Strong application. All risk thresholds met. Proceed to final verification."),
    (0.45, 0.70, "MANUAL REVIEW",
     "Borderline profile. Additional documentation and underwriter review required."),
    (0.00, 0.45, "REJECTED",
     "High-risk profile. Application does not meet minimum approval criteria."),
]

CATEGORY_COLOURS = {
    "APPROVED":       "#10b981",
    "MANUAL REVIEW":  "#f59e0b",
    "REJECTED":       "#ef4444",
}

CATEGORY_ICONS = {
    "APPROVED":       "✅",
    "MANUAL REVIEW":  "⚠️",
    "REJECTED":       "❌",
}


def _load_artefacts(model_path: str = "models/loannet_best.h5",
                    scaler_path: str = "models/scaler.pkl",
                    encoder_path: str = "models/encoders.pkl") -> None:
    global _model, _scaler, _encoders
    _model    = keras.models.load_model(model_path)
    _scaler   = joblib.load(scaler_path)
    _encoders = joblib.load(encoder_path)


def _encode(applicant: dict) -> dict:
    """Label-encode categorical fields using saved encoders."""
    a = dict(applicant)
    for col, enc in (_encoders or {}).items():
        if col in a:
            val = str(a[col])
            if val not in set(enc.classes_):
                val = enc.classes_[0]   # fallback to first known class
            a[col] = int(enc.transform([val])[0])
    return a


def predict_applicant(applicant: dict,
                      feature_order: list = None,
                      threshold_approve: float = 0.70,
                      threshold_review:  float = 0.45) -> dict:
    """
    Predict loan approval for a single applicant.

    Args:
        applicant        : dict mapping feature names to values
        feature_order    : ordered list of feature names (from training)
        threshold_approve: probability above which loan is auto-approved
        threshold_review : probability above which loan goes to manual review

    Returns:
        dict with keys:
            decision    - "APPROVED" | "MANUAL REVIEW" | "REJECTED"
            probability - approval probability as percentage (0–100)
            risk_score  - rejection risk as percentage (0–100)
            colour      - hex colour for the decision
            icon        - emoji icon
            advice      - explanatory string
    """
    global _model, _scaler, _encoders

    if _model is None:
        try:
            _load_artefacts()
        except Exception as e:
            return {"error": f"Model not loaded: {e}"}

    a = _encode(applicant)

    if feature_order:
        x = np.array([[a.get(f, 0.0) for f in feature_order]], dtype=np.float32)
    else:
        x = np.array([[v for v in a.values()]], dtype=np.float32)

    x_sc = _scaler.transform(x)
    prob = float(_model.predict(x_sc, verbose=0)[0][0])
    prob = float(np.clip(prob, 0.0, 1.0))

    # Map to decision
    decision = "REJECTED"
    advice   = DECISIONS[-1][3]
    for lo, hi, dec, adv in DECISIONS:
        if lo <= prob < hi:
            decision, advice = dec, adv
            break

    return {
        "decision":    decision,
        "probability": round(prob * 100, 1),
        "risk_score":  round((1 - prob) * 100, 1),
        "colour":      CATEGORY_COLOURS[decision],
        "icon":        CATEGORY_ICONS[decision],
        "advice":      advice,
    }


def batch_predict(df_applicants, feature_order: list = None) -> list:
    """
    Predict for a pandas DataFrame of applicants.
    Returns list of result dicts.
    """
    return [predict_applicant(row.to_dict(), feature_order)
            for _, row in df_applicants.iterrows()]


if __name__ == "__main__":
    # ── Example usage ────────────────────────────────────────
    sample_applicant = {
        "age":              35,
        "income":           75_000,
        "loan_amount":      200_000,
        "loan_term":        360,
        "credit_score":     720,
        "employment_years": 7.0,
        "debt_to_income":   0.28,
        "num_credit_lines": 5,
        "num_delinquencies":0,
        "property_value":   350_000,
        "loan_to_income":   2.67,
        "loan_to_value":    0.57,
        "monthly_payment":  1_000,
        "savings_ratio":    0.30,
        "education":        "Bachelor",
        "employment_type":  "Salaried",
        "loan_purpose":     "Home",
    }

    result = predict_applicant(sample_applicant)
    print(f"\n{result['icon']} Decision:    {result['decision']}")
    print(f"   Probability: {result['probability']}%")
    print(f"   Risk Score:  {result['risk_score']}%")
    print(f"   Advice:      {result['advice']}")
