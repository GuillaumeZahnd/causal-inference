from re import I

import numpy as np
from sklearn.metrics import mean_squared_error
from causal_parameters import CausalParameters


def bootstrap_cate_ci(method, df, feature_columns, nb_boots=200, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_estimates = np.empty((nb_boots, n))
    for b in range(nb_boots):
        idx = rng.integers(0, n, n)
        boot_df = df.iloc[idx].reset_index(drop=True)
        boot_estimates[b] = method(boot_df, feature_columns)
    lower = np.percentile(boot_estimates, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def print_metrics(
    method_name: str,
    causal_parameters: CausalParameters,
    true_cate: np.ndarray | None = None,
    ) -> None:

    predicted_cate = causal_parameters.cate

    if causal_parameters.ate is not None:
        predicted_ate = causal_parameters.ate
    else:
        predicted_ate = np.mean(predicted_cate)

    print("="*64)
    print(method_name)

    # Synthetic data with ground truth available
    if true_cate is not None:
        true_ate = float(np.mean(true_cate))
        ate_bias = predicted_ate - true_ate
        mse_cate = mean_squared_error(true_cate, predicted_cate)

        print(f"True ATE: {true_ate:.2f}")

        if causal_parameters.ate_lower is not None and causal_parameters.ate_upper is not None:
            ate_lower = causal_parameters.ate_lower
            ate_upper = causal_parameters.ate_upper
            ate_ci = f", 95% CI: [{ate_lower:.2f}, {ate_upper:.2f}], Width: {ate_upper - ate_lower:.2f}"
        else:
            ate_ci = ""
        print(f"Predicted ATE: {predicted_ate:.2f}, Bias: {ate_bias:.2f}{ate_ci}")

        print(f"CATE MSE: {mse_cate:.2f}")

        # Print CATE CI summary and coverage rate if available
        if causal_parameters.cate_lower is not None and causal_parameters.cate_upper is not None:
            cate_lower = causal_parameters.cate_lower
            cate_upper = causal_parameters.cate_upper

            avg_width = float(np.mean(cate_upper - cate_lower))
            coverage_rate = float(np.mean((true_cate >= cate_lower) & (true_cate <= cate_upper)))

            print(
                f"Mean CATE 95% CI: [{np.mean(cate_lower):.2f}, {np.mean(cate_upper):.2f}], "
                f"Average width: {avg_width:.2f}, "
                f"Coverage rate: {coverage_rate:.1%}"
                )


    # Observed data with unavailable ground truth
    else:
        if causal_parameters.ate_lower is not None and causal_parameters.ate_upper is not None:
            ate_lower, ate_upper = causal_parameters.ate_lower, causal_parameters.ate_upper
            ate_ci = f", 95% CI: [{ate_lower:.2f}, {ate_upper:.2f}], Width: {ate_upper - ate_lower:.2f}"
        else:
            ate_ci = ""
        print(f"Predicted ATE: {predicted_ate:.2f}{ate_ci}")

        # Measure estimated treatment effect variability across units
        cate_std = float(np.std(predicted_cate))
        q25, q50, q75 = np.percentile(predicted_cate, [25, 50, 75])

        print(f"CATE Distribution — Mean: {np.mean(predicted_cate):.2f}, Std: {cate_std:.2f}")
        print(f"CATE Quartiles — 25th: {q25:.2f}, Median: {q50:.2f}, 75th: {q75:.2f}")

        if causal_parameters.cate_lower is not None and causal_parameters.cate_upper is not None:
            cate_lower, cate_upper = causal_parameters.cate_lower, causal_parameters.cate_upper
            avg_width = float(np.mean(cate_upper - cate_lower))
            print(
                f"Mean CATE 95% CI: [{np.mean(cate_lower):.2f}, {np.mean(cate_upper):.2f}], "
                f"Average width: {avg_width:.2f}"
                )

    print("")
