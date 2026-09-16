import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from utils import print_metrics



def xgboost_s_learner(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> np.ndarray:
    """
    XGBoost S-Learner (single model)
    """

    X = df[feature_columns]
    W = df["treatment"]
    Y = df["outcome"]

    # Include treatment (W) directly as a feature
    X_s = X.copy()
    X_s["treatment"] = W

    s_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    s_model.fit(X_s, Y)

    # Predict counterfactuals by setting treatment explicitly to 1 and 0
    X_counter_1 = X.copy().assign(treatment=1)
    X_counter_0 = X.copy().assign(treatment=0)

    predicted_cate = s_model.predict(X_counter_1) - s_model.predict(X_counter_0)

    return predicted_cate


def xgboost_t_learner(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> np.ndarray:
    """
    XGBoost T-Learner (dual model)
    """

    X = df[feature_columns]
    W = df["treatment"]
    Y = df["outcome"]

    # Split data by treatment arm
    X_0, Y_0 = X[W == 0], Y[W == 0]
    X_1, Y_1 = X[W == 1], Y[W == 1]

    t_model_0 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    t_model_1 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)

    t_model_0.fit(X_0, Y_0)
    t_model_1.fit(X_1, Y_1)

    # Predict counterfactuals across the entire dataset X
    predicted_cate = t_model_1.predict(X) - t_model_0.predict(X)

    return predicted_cate
