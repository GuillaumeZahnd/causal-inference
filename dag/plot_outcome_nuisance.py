import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_outcome_nuisance(y_pred, Y, treatment, residuals, r2, rmse):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground-truth vs predicted outcomes
    sns.scatterplot(
        x=y_pred,
        y=Y,
        hue=treatment.map({1: "Treated", 0: "Control"}),
        alpha=0.4,
        ax=axes[0],
    )

    axes[0].axline(
        (0, 0), slope=1, color="black", linestyle="--", label="Ideal fit"
    )
    axes[0].set_title(rf"Outcome fit ($R^2 = {r2:.3f}$)")
    axes[0].set_xlabel(r"Predicted outcome $\hat{m}(X)$")
    axes[0].set_ylabel(r"Ground truth outcome $Y$")

    # Residual density by treatment group
    plot_df = pd.DataFrame({
        "residual": residuals,
        "Group": treatment.map({1: "Treated", 0: "Control"}),
    })

    sns.histplot(
        data=plot_df,
        x="residual",
        hue="Group",
        stat="density",
        common_norm=False,
        kde=True,
        ax=axes[1],
        alpha=0.4,
        kde_kws={"cut": 0},
    )

    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_title(
        rf"Outcome residual distribution ($Y - \hat{{m}}(X)$)"
        + f"\nRMSE = {rmse:.3f}"
    )
    axes[1].set_xlabel("Residual value")

    # Bland-Altman plot
    mean_val = (Y + y_pred) / 2
    diff_val = Y - y_pred  # Residuals: Y - m_hat(X)

    mean_diff = np.mean(diff_val)
    sd_diff = np.std(diff_val, ddof=1)
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    sns.scatterplot(
        x=mean_val,
        y=diff_val,
        hue=treatment.map({1: "Treated", 0: "Control"}),
        alpha=0.4,
        ax=axes[2],
    )

    axes[2].axhline(
        mean_diff, color="black", linestyle="-", label=f"Mean ({mean_diff:.2f})"
    )
    axes[2].axhline(
        upper_loa,
        color="black",
        linestyle="--",
        label=f"+1.96 SD ({upper_loa:.2f})",
    )
    axes[2].axhline(
        lower_loa,
        color="black",
        linestyle="--",
        label=f"-1.96 SD ({lower_loa:.2f})",
    )

    axes[2].set_title("Bland-Altman Plot")
    axes[2].set_xlabel(r"Mean of outcome and prediction $\frac{Y + \hat{m}(X)}{2}$")
    axes[2].set_ylabel(r"Difference $Y - \hat{m}(X)$")
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
