import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from xgboost import XGBRegressor


def double_ml_econml(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> np.ndarray:

    X = df[feature_columns]
    W = df["treatment"]
    Y = df["outcome"]

    # Define base machine learning model for outcome nuisance: m(X) = E[Y | X]
    model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)

    # Define base machine learning model for propensity nuisance: e(X) = P(W=1 | X)
    model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)

    # Initialize the double machine learning estimator
    dml_estimator = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        cv=5,  # Uses 5-fold cross-fitting natively to prevent nuisance overfitting bias
        random_state=0,
        )

    # Fit the double ML model on observational data [X, W, Y] (EconML expects Y and W as 1D arrays or column vectors)
    dml_estimator.fit(Y=Y.to_numpy(), T=W.to_numpy(), X=X)

    # Predict counterfactual treatment effects (CATE) across sample X
    predicted_cate_raw = dml_estimator.effect(X)

    # Flatten output array to 1D float64 array for evaluation
    predicted_cate = np.asarray(predicted_cate_raw, dtype=np.float64).ravel()

    return predicted_cate
