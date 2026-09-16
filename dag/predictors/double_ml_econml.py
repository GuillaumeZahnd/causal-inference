import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from xgboost import XGBRegressor

from causal_parameters import CausalParameters


def double_ml_econml(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> CausalParameters:

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

    # Confidence intervals for the quantities \tau(X, T0, T1) produced by the model (one pair per subject)
    cate_lower, cate_upper = dml_estimator.effect_interval(X, alpha=0.05)

    # Confidence interval for the quantity E[\tau(X, T0, T1)] produced by the model (one pair in total)
    ate = float(dml_estimator.ate(X))
    ate_lower, ate_upper = dml_estimator.ate_interval(X)

    # Predict counterfactual treatment effects (CATE) across sample X
    cate_raw = dml_estimator.effect(X)

    # Flatten output array to 1D float64 array for evaluation
    cate = np.asarray(cate_raw, dtype=np.float64).ravel()

    causal_parameters = CausalParameters(
        cate=cate,
        cate_lower=cate_lower,
        cate_upper=cate_upper,
        ate=ate,
        ate_lower=ate_lower,
        ate_upper=ate_upper
        )

    return causal_parameters
