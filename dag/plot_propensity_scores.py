import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def compute_overlap_coefficient(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    nb_bins: int = 100,
) -> float:

    e_1 = propensity_scores[treatment == 1]
    e_0 = propensity_scores[treatment == 0]

    # Shared bin edges bounded between 0 and 1
    bins = np.linspace(0, 1, nb_bins + 1)

    # Compute normalized probability densities for each group
    histogram_1, _ = np.histogram(e_1, bins=bins, density=True)
    histogram_0, _ = np.histogram(e_0, bins=bins, density=True)

    # Bin width
    dx = 1.0 / nb_bins

    # Numerical integration of overlapping area
    overlap_coefficient = float(np.sum(np.minimum(histogram_1, histogram_0)) * dx)

    return overlap_coefficient


def plot_propensity_scores(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment_column: str = "treatment",
    nb_bins: int = 50
) -> None:

    overlap_coefficient = compute_overlap_coefficient(
        propensity_scores=propensity_scores,
        treatment=df[treatment_column].to_numpy(),
        nb_bins=nb_bins,
    )

    data_with_label_mapping = df.assign(
        propensity=propensity_scores,
        Treatment=df[treatment_column].replace({1: "Treated", 0: "Control"}),
    )

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=data_with_label_mapping,
        x="propensity",
        hue="Treatment",
        stat="density",
        common_norm=False,
        kde=False,
        binrange=(0, 1),
        bins=nb_bins,
        alpha=0.5,
    )

    plt.title(f"Propensity score distribution by treatment group\nOverlap coefficient: {overlap_coefficient:.2f}", fontsize=12)
    plt.xlabel(r"$e(X) := P(T=1 \mid X)$", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.xlim(0, 1)
    plt.show()
