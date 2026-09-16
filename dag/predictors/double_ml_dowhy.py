import numpy as np
import pandas as pd
from dowhy import CausalModel
from xgboost import XGBRegressor

from causal_parameters import CausalParameters


def double_ml_dowhy(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> CausalParameters:

    X = df[feature_columns].values

    model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)

    # Map features only to effect_modifiers to target EconML's X parameter.
    # We omit common_causes so W remains empty natively, avoiding duplication.
    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="outcome",
        common_causes=feature_columns,
        effect_modifiers=feature_columns,
        )

    # Identify estimand (proceed=True allows it to pass without explicit common_causes)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Leave fit_params empty to avoid keyword collisions with DoWhy's internal wrapper
    estimate = model.estimate_effect(
        identified_estimand=identified_estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        method_params={
            "init_params": {
                "model_y": model_y,
                "model_t": model_t,
                "cv": 5,
                "random_state": 0,
                },
            "fit_params": {},  # Must be empty
            },
        )

    # Extract unit-level predictions from the underlying estimator
    econml_estimator = estimate.estimator.estimator
    cate_raw = econml_estimator.effect(df[feature_columns].values)

    cate = np.asarray(cate_raw, dtype=np.float64).ravel()

    cate_lower_raw, cate_upper_raw = econml_estimator.effect_interval(X, alpha=0.05)
    cate_lower = np.asarray(cate_lower_raw, dtype=np.float64).ravel()
    cate_upper = np.asarray(cate_upper_raw, dtype=np.float64).ravel()

    # ATE point estimate and confidence intervals
    ate = float(econml_estimator.ate(X))
    ate_lower_raw, ate_upper_raw = econml_estimator.ate_interval(X, alpha=0.05)
    ate_lower = float(ate_lower_raw)
    ate_upper = float(ate_upper_raw)

    causal_parameters = CausalParameters(
        cate=cate,
        cate_lower=cate_lower,
        cate_upper=cate_upper,
        ate=ate,
        ate_lower=ate_lower,
        ate_upper=ate_upper
        )

    return causal_parameters
