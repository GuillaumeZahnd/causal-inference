import numpy as np
import pandas as pd
from dowhy import CausalModel
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from causal_parameters import CausalParameters


def double_ml_dowhy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> CausalParameters:

    random_seed = 0
    sorted_features = sorted(feature_columns)

    # Deterministic cross-validation splitter
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=1, random_state=random_seed)
    model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=1, random_state=random_seed)

    # Pass sorted_features to minimize set ordering drift
    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="outcome",
        effect_modifiers=sorted_features,
    )

    # Identify estimand (proceed=True allows it to pass without explicit common_causes)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        identified_estimand=identified_estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        method_params={
            "init_params": {
                "model_y": model_y,
                "model_t": model_t,
                "cv": cv_splitter,
                "n_estimators": 100,
                "n_jobs": 1,
                "random_state": random_seed,
            },
            "fit_params": {},
        },
    )

    econml_estimator = estimate.estimator.estimator

    # Extract the exact column sequence DoWhy passed to fit()
    fitted_feature_names = estimate.estimator._effect_modifier_names
    X_eval = df[fitted_feature_names].values

    cate_raw = econml_estimator.effect(X_eval)
    cate = np.asarray(cate_raw, dtype=np.float64).ravel()

    cate_lower_raw, cate_upper_raw = econml_estimator.effect_interval(X_eval, alpha=0.05)
    cate_lower = np.asarray(cate_lower_raw, dtype=np.float64).ravel()
    cate_upper = np.asarray(cate_upper_raw, dtype=np.float64).ravel()

    # ATE point estimate and confidence intervals
    ate = float(econml_estimator.ate(X_eval))
    ate_lower_raw, ate_upper_raw = econml_estimator.ate_interval(X_eval, alpha=0.05)
    ate_lower = float(ate_lower_raw)
    ate_upper = float(ate_upper_raw)

    causal_parameters = CausalParameters(
        cate=cate,
        cate_lower=cate_lower,
        cate_upper=cate_upper,
        ate=ate,
        ate_lower=ate_lower,
        ate_upper=ate_upper,
    )

    return causal_parameters
