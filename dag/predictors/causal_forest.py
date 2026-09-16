import numpy as np
import pandas as pd
from econml.grf import CausalForest

from causal_parameters import CausalParameters


def causal_forest(
    df: pd.DataFrame,
    feature_columns: list[str],
    ) -> CausalParameters:

    X = df[feature_columns]
    W = df["treatment"]
    Y = df["outcome"]

    # Initialize the causal forest estimator
    cf_estimator = CausalForest(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=None,
        random_state=0,
        )

    # Fit the causal forest on observational data [X, W, Y] (GRF expects T and Y as 1D arrays)
    cf_estimator.fit(X, W.to_numpy(), Y.to_numpy())

    # Predict counterfactual treatment effects (CATE) across sample X
    cate_raw = cf_estimator.predict(X)

    # Flatten output array to 1D float64 array for evaluation
    cate = np.asarray(cate_raw, dtype=np.float64).ravel()

    causal_parameters = CausalParameters(cate=cate)

    return causal_parameters
