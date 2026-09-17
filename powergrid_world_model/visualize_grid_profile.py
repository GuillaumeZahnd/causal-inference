import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from environment import generate_synthetic_day_data



def plot_daily_power_profile(df: pd.DataFrame) -> None:

    """
    Plot time-of-day against solar yield and household demand load.

    Args:
        df: DataFrame containing the time, solar, and demand data.
    """

    sns.set_theme(style="whitegrid", font="sans-serif")

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")

    solar_color = "#E66101"
    demand_color = "#2B83BA"

    # --- Plot Solar Generation (Line + Shaded Area) ---
    ax.plot(
        df["hour"],
        df["solar_yield"],
        color=solar_color,
        linewidth=2.5,
        label="Solar yield (kW)",
    )
    ax.fill_between(
        df["hour"], df["solar_yield"], alpha=0.25, color=solar_color
    )

    # --- Plot Demand Load (Line + Shaded Area) ---
    ax.plot(
        df["hour"],
        df["demand_load"],
        color=demand_color,
        linewidth=2.5,
        linestyle="--",
        label="Demand load (kW)",
    )
    ax.fill_between(
        df["hour"], df["demand_load"], alpha=0.15, color=demand_color
    )

    # Formatting axes and ticks
    ax.set_xlabel("Time of day (Hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Power (kW)", fontsize=11, fontweight="bold")
    ax.set_title("Daily grid profile: Solar yield vs. Demand load", fontsize=13, fontweight="bold", pad=12)

    # Set exact 24-hour ticks
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

    # Clean aesthetic polish
    sns.despine(ax=ax)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    plt.savefig("daily_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = generate_synthetic_day_data(steps_per_hour=4)
    plot_daily_power_profile(df=df)
