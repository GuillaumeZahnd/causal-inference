import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from environment import generate_synthetic_day_data


def plot_daily_power_profile(df: pd.DataFrame) -> None:

    sns.set_theme(style="whitegrid", font="sans-serif")

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")

    solar_color = "#E66101"
    demand_color = "#2B83BA"
    price_color = "#4DAC26"

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

    # Formatting axes and ticks
    ax.set_xlabel("Time of day (Hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Power (kW)", fontsize=11, fontweight="bold")
    ax.set_title("Daily grid profile: Solar yield vs. Demand load vs. Spot price", fontsize=13, fontweight="bold", pad=12)

    # Set exact 24-hour ticks
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

    # Spot price on secondary axis
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

    # Clean aesthetic polish
    sns.despine(ax=ax, right=False)
    sns.despine(ax=ax2, left=True)

    # Combine legends from both axes into one
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    plt.savefig("daily_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = generate_synthetic_day_data(steps_per_hour=4)
    plot_daily_power_profile(df=df)
