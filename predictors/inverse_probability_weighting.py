import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def inverse_probability_weighting(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> np.ndarray:

    X = df[feature_columns].values
    T = df["treatment"].values.astype(int)
    Y = df["outcome"].values.astype(float)

    # Fit propensity score model P(T=1 | X)
    propensity_model = LogisticRegression(random_state=0)
    propensity_model.fit(X, T)

    # Predict propensity scores e(X) and clip to avoid division by zero or extreme weights
    propensity_scores = propensity_model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

    # Compute inverse propensity weights
    weights = np.where(T == 1, 1.0 / propensity_scores, 1.0 / (1.0 - propensity_scores))

    # Calculate Hajek stabilized ATE via weighted means
    weighted_mean_treated = np.sum(Y * weights * T) / np.sum(weights * T)
    weighted_mean_control = np.sum(Y * weights * (1 - T)) / np.sum(weights * (1 - T))

    ate_scalar = float(weighted_mean_treated - weighted_mean_control)

    # Expand ATE scalar to a 1D array matching true_cate length for evaluation
    predicted_cate = np.full(df.shape[0], ate_scalar, dtype=np.float64)

    return predicted_cate

