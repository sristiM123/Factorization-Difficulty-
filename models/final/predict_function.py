
def predict_rsa_vulnerability(N: int) -> dict:
    """
    Predict RSA semiprime vulnerability from public modulus alone.

    Parameters
    ----------
    N : int
        RSA public modulus (semiprime N = p * q)

    Returns
    -------
    dict : Risk assessment with keys:
        - vulnerability_percentile
        - best_attack_algorithm
        - attack_confidence
        - migration_tier
        - pqc_recommendation
    """
    import joblib
    import numpy as np
    from src.features import extract_features

    # Load models (cache in production)
    rf = joblib.load("models/final/random_forest_regressor.pkl")
    cls = joblib.load("models/final/calibrated_classifier.pkl")
    le = joblib.load("models/final/label_encoder.pkl")

    # Extract features
    features = extract_features([N], verbose=False)

    # Predict
    pred_runtime = rf.predict(features)[0]
    pred_proba = cls.predict_proba(features)[0]
    pred_algo = le.classes_[np.argmax(pred_proba)]
    confidence = np.max(pred_proba) * 100

    return {
        "N": N,
        "bit_length": N.bit_length(),
        "predicted_log2_runtime": round(pred_runtime, 3),
        "best_attack": pred_algo,
        "confidence": round(confidence, 1),
        "migration_tier": "URGENT" if pred_runtime < -10 else
                          "PLANNED" if pred_runtime < -5 else "MONITOR",
    }
