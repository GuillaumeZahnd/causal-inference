from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
from typing import cast

from environment import generate_synthetic_day_data


def plot_exogenous_factors(
    df: pd.DataFrame,
    fig: Figure,
    ax: Axes,
) -> tuple[Figure, Axes]:
    """Plot solar yield, demand load, and spot price."""
    sns.set_theme(style="whitegrid", font="sans-serif")

    solar_color = "#E66101"
    demand_color = "#2B83BA"
    price_color = "#4D8C26"

    # Solar yield
    ax.plot(
        df["hour"],
        df["solar_yield"],
        color=solar_color,
        linewidth=2.5,
        label="Solar yield (kW)",
    )
    ax.fill_between(df["hour"], df["solar_yield"], alpha=0.25, color=solar_color)

    # Demand load
    ax.plot(
        df["hour"],
        df["demand_load"],
        color=demand_color,
        linewidth=2.5,
        linestyle="--",
        label="Demand load (kW)",
    )
    ax.fill_between(df["hour"], df["demand_load"], alpha=0.15, color=demand_color)

    # Axes styling
    ax.set_ylabel("Power (kW)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Grid & Environment Profile: Solar, Demand, Spot Price",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    max_hour = int(cast(int, df["hour"].max()))
    ax.set_xlim(0, max_hour)
    ax.set_xticks(range(0, max_hour + 1, 2 if max_hour <= 48 else 6))

    # Spot price (secondary axis)
    ax2 = ax.twinx()
    ax2.plot(
        df["hour"],
        df["spot_price"],
        color=price_color,
        linewidth=2.0,
        linestyle=":",
        label="Spot price (€/kWh)",
    )
    ax2.set_ylabel("Spot price (€/kWh)", fontsize=11, fontweight="bold", color=price_color)
    ax2.tick_params(axis="y", labelcolor=price_color)
    ax2.grid(False)

    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        loc="upper left",
    )

    return fig, ax


if __name__ == "__main__":
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exogenous_factors.png"

    df = generate_synthetic_day_data(steps_per_hour=4)

    fig, ax = plt.subplots(1, 1, figsize=(13, 4), sharex=True, layout="constrained")
    plot_exogenous_factors(df=df, fig=fig, ax=ax)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
