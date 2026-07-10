import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
except Exception as e:
    raise ImportError("shap is required for tabular explanations. Install with `pip install shap`")

from multimodal.tabular_ensemble import ensemble_tabular


def explain_tabular(tabular_models: dict, X_instance, background=None, nsamples=50):
    """Return a matplotlib Figure with SHAP explanations for the ensemble prediction.

    - `tabular_models`: dict of fitted sklearn-like models
    - `X_instance`: a pandas.DataFrame or 2D array with a single instance (1, n_features)
    - `background`: optional background array for KernelExplainer
    - `nsamples`: KernelExplainer nsamples (lower is faster)
    """
    import pandas as pd

    # Ensure X_instance is a DataFrame so shap plotting labels work
    if isinstance(X_instance, pd.DataFrame):
        X_df = X_instance
        x_np = X_df.values
    else:
        X_df = pd.DataFrame(X_instance)
        x_np = X_df.values

    def ensemble_fn(X):
        preds = []
        for row in X:
            row_df = pd.DataFrame([row], columns=X_df.columns)
            p, _ = ensemble_tabular(tabular_models, row_df)
            preds.append(p)
        return np.array(preds)

    if background is None:
        background = np.zeros((1, x_np.shape[1]))

    explainer = shap.KernelExplainer(ensemble_fn, background)
    shap_values = explainer.shap_values(x_np, nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    values = shap_values[0] if shap_values.ndim == 2 else shap_values
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    explanation = shap.Explanation(values=values,
                                   base_values=base_value,
                                   data=x_np[0],
                                   feature_names=list(X_df.columns))

    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    return fig


def explain_tabular_table(tabular_models: dict, X_instance, background=None, nsamples=50):
    """Return a pandas DataFrame of SHAP values for the ensemble prediction."""
    import pandas as pd

    if isinstance(X_instance, pd.DataFrame):
        X_df = X_instance
        x_np = X_df.values
    else:
        X_df = pd.DataFrame(X_instance)
        x_np = X_df.values

    def ensemble_fn(X):
        preds = []
        for row in X:
            row_df = pd.DataFrame([row], columns=X_df.columns)
            p, _ = ensemble_tabular(tabular_models, row_df)
            preds.append(p)
        return np.array(preds)

    if background is None:
        background = np.zeros((1, x_np.shape[1]))

    explainer = shap.KernelExplainer(ensemble_fn, background)
    shap_values = explainer.shap_values(x_np, nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": X_df.columns,
        "shap_value": shap_values[0] if shap_values.ndim > 1 else shap_values
    })
    return shap_df
