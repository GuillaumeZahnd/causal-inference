import numpy as np
from sklearn.metrics import mean_squared_error


def print_metrics(
    method_name: str,
    predicted_cate: np.ndarray,
    true_cate: np.ndarray
    ) -> None:

    mse_cate = mean_squared_error(true_cate, predicted_cate)
    predicted_ate = np.mean(predicted_cate)
    true_ate = np.mean(true_cate)
    ate_bias = predicted_ate - true_ate

    print("="*64)
    print(method_name)
    print(f"CATE MSE: {mse_cate:.4f}")
    print(f"Predicted ATE: {predicted_ate:.4f}")
    print(f"True ATE: {true_ate:.4f}")
    print(f"ATE bias: {ate_bias:.4f}")
    print("")
