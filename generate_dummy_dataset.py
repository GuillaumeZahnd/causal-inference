from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import os


@dataclass
class DAGParams:
    nb_subjects: int = 10_000
    seed: int = 0
    include_hidden_confounder: bool = False
    noise_std: float = 5.0


def _propensity_logit(age: np.ndarray, severity: np.ndarray, comorbidity: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Log-odds of treatment under the true data-generating process."""
    return (
        -4.0
        + 0.04 * age
        + 0.35 * severity
        + 0.45 * comorbidity
        + 0.01 * (age - 55) * severity / 10  # Nonlinearity
        + 1.2 * u  # Only active if hidden confounder is switched on
    )


def _baseline_outcome(age: np.ndarray, severity: np.ndarray, comorbidity: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Y0: potential outcome under no treatment, on a recovery-score-like scale."""
    return (
        70
        - 0.25 * age
        - 3.0 * severity
        - 2.0 * comorbidity
        + 0.02 * age * severity / 10  # nonlinearity
        - 4.0 * u  # Only active if hidden confounder is switched on
    )


def _individual_treatment_effect(age: np.ndarray, severity: np.ndarray) -> np.ndarray:
    """
    Conditional Average Treatment Effect (CATE)
    CATE(x): true treatment effect, heterogeneous by age and severity:
        - Baseline effect: 8.0 points for a subject at age 55 with severity 0.
        - Linear age effect: -0.08 per year relative to age 55.
        - Quadratic severity effect: +0.3 per square unit of severity.
    """
    return 8.0 - 0.08 * (age - 55) + 0.3 * severity ** 2


def generate_data(params: DAGParams = DAGParams()) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
        df: observed dataset an estimator would actually have access to
            (e.g., age, severity, comorbidity, treatment, outcome)
        ground_truth: dict of non-observed values only known by the simulator
            (e.g., Y0, Y1, true_cate, true_ate, propensity, and U if hidden)
    """
    rng = np.random.default_rng(params.seed)
    nb_subjects = params.nb_subjects

    age = rng.uniform(30, 81, nb_subjects)  # Between 30 and 80 y.o.
    severity = rng.integers(0, 11, nb_subjects)  # In a scale between 0 and 10
    comorbidity = rng.poisson(1.0, nb_subjects)
    u = rng.normal(0, 1, nb_subjects) if params.include_hidden_confounder else np.zeros(nb_subjects)

    # Convert the logits to a value in ]0, 1[
    propensity = 1 / (1 + np.exp(-_propensity_logit(age, severity, comorbidity, u)))

    treatment = rng.binomial(n=1, p=propensity)

    eps = rng.normal(0, params.noise_std, nb_subjects)  # shared noise -> same draw feeds Y0 and Y1

    # Counterfactual potential outcome under control (scenario in which subjects do not receive the treatment)
    y0 = _baseline_outcome(age, severity, comorbidity, u) + eps

    # Conditional Average Treatment Effect (CATE)
    cate = _individual_treatment_effect(age, severity)

    # Counterfactual potential outcome under treatment (scenario in which subjects receive the treatment)
    y1 = y0 + cate

    outcome = np.where(treatment == 1, y1, y0)

    df = pd.DataFrame({
        "age": age,
        "severity": severity,
        "comorbidity": comorbidity,
        "treatment": treatment,
        "outcome": outcome,
    })

    ground_truth = {
        "y0": y0,
        "y1": y1,
        "true_cate": cate,
        "true_ate": float(np.mean(cate)),
        "propensity": propensity,
        "u": u,
    }
    return df, ground_truth


def diagnostics(df: pd.DataFrame, ground_truth: dict) -> None:
    n_treated = df["treatment"].sum()
    pct_treated = df["treatment"].mean()
    true_ate = ground_truth["true_ate"]

    y_tr = df.loc[df.treatment == 1, "outcome"].mean()
    y_ctrl = df.loc[df.treatment == 0, "outcome"].mean()
    naive_difference = y_tr - y_ctrl

    print(f"Subjects: {len(df):,} | Treated: {n_treated:,} ({pct_treated:.1%})")
    print(f"True ATE: {true_ate:.3f} | Naive difference: {naive_difference:.3f} | Bias: {naive_difference - true_ate:.3f}\n")

    print("Covariate Means by Group:")
    print(df.groupby("treatment")[["age", "severity", "comorbidity"]].mean().round(2))

    p = ground_truth["propensity"]
    print(f"\nPropensity Range: [{p.min():.3f}, {p.max():.3f}]")


if __name__ == "__main__":

    dag_params = DAGParams(
        nb_subjects=10_000,
        seed=0,
        include_hidden_confounder=True,
        noise_std=10.0
        )

    df, ground_truth = generate_data(params=dag_params)

    diagnostics(df, ground_truth)

    print("\nObserved dataset")
    print(df.describe().round(2))
    print(df.head().round(2))

    # Save to disk
    output_dir = Path("datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dummy_observed_dataset.csv"
    npz_path = output_dir / "dummy_ground_truth.npz"
    
    df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **ground_truth)
    
    print(f"\nSuccessfully saved '{csv_path}' and '{npz_path}' to disk.")
