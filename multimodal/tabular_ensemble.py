import numpy as np

# F1 scores from validation (fixed constants)
TABULAR_F1 = {
    "Logistic Regression": 0.892,
    "Random Forest": 0.899,
    "SVM": 0.899,
    "XGBoost": 0.903
}

def ensemble_tabular(tabular_models: dict, X):
    """
    Returns a single clinically meaningful probability:
    P(Heart Failure | Tabular Data)
    """
    probs = []
    weights = []

    for name, model in tabular_models.items():
        p = float(model.predict_proba(X)[0, 1])
        probs.append(p)
        weights.append(TABULAR_F1[name])

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    P_tab = float(np.sum(weights * np.array(probs)))

    return P_tab, dict(zip(tabular_models.keys(), probs))
