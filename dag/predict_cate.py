from pathlib import Path
import numpy as np
import pandas as pd
from predictors.inverse_probability_weighting import inverse_probability_weighting
from predictors.xgboost_s_t_learners import xgboost_s_learner, xgboost_t_learner
from predictors.double_ml_econml import double_ml_econml
from predictors.double_ml_dowhy import double_ml_dowhy
from predictors.causal_forest import causal_forest

from utils import print_metrics


if __name__ == "__main__":

    csv_path = Path("datasets") / "dummy_observed_dataset.csv"
    npz_path = Path("datasets") / "dummy_ground_truth.npz"

    ground_truth_file = np.load(npz_path)
    ground_truth = {key: ground_truth_file[key] for key in ground_truth_file.files}
    true_cate = ground_truth["true_cate"]

    df = pd.read_csv(csv_path)
    feature_columns = ["age", "severity", "comorbidity"]


    METHODS = [
        ("Inverse Probability Weighting", inverse_probability_weighting),
        ("XGBoost S-Learner (single model)", xgboost_s_learner),
        ("XGBoost T-Learner (dual model)", xgboost_t_learner),
        ("Causal Forest", causal_forest),
        ("Double ML (EconML)", double_ml_econml),
        ("Double ML (DoWhy)", double_ml_dowhy),
    ]

    for name, method in METHODS:
        causal_parameters = method(df, feature_columns)
        print_metrics(name, causal_parameters, true_cate)

