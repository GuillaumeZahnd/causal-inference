import numpy as np
import pandas as pd
from dowhy import CausalModel
from xgboost import XGBRegressor


def double_ml_dowhy(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> np.ndarray:

    model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)

    # Map features only to effect_modifiers to target EconML's X parameter.
    # We omit common_causes so W remains empty natively, avoiding duplication.
    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="outcome",
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
    predicted_cate_raw = econml_estimator.effect(df[feature_columns].values)

    predicted_cate = np.asarray(predicted_cate_raw, dtype=np.float64).ravel()

    return predicted_cate
