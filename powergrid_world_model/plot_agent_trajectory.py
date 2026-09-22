import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path

from agent import BaseAgent, RandomAgent, SB3Agent
from battery import Battery
from environment import Environment
from plot_exogenous_factors import plot_exogenous_factors


def plot_agent_summary(
    df: pd.DataFrame, ax: Axes | None = None
) -> tuple[Figure, Axes, Axes]:
    """Plots agent battery actions (charge/discharge), State of Charge (SoC), and cumulative rewards."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    else:
        fig = ax.get_figure()

    soc_color = "#7570B3"
    charge_color = "#1B9E77"
    discharge_color = "#D95F02"
    cum_reward_color = "#E7298A"

    # Action bars (Charge > 0, Discharge < 0)
    charges = np.maximum(df["action_kw"], 0)
    discharges = np.minimum(df["action_kw"], 0)

    ax.bar(
        df["hour"],
        charges,
        width=0.08,
        color=charge_color,
        alpha=0.7,
        label="Charge (kW)",
    )
    ax.bar(
        df["hour"],
        discharges,
        width=0.08,
        color=discharge_color,
        alpha=0.7,
        label="Discharge (kW)",
    )

    # State of Charge line
    ax.plot(
        df["hour"],
        df["battery_soc"],
        color=soc_color,
        linewidth=2.5,
        label="Battery SoC (kWh)",
    )

    ax.set_xlabel("Time (Hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Battery power (kW) / SoC (kWh)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Agent dispatch summary: Actions, SoC & Cumulative reward",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    max_hour = int(df["hour"].max())
    ax.set_xlim(0, max_hour)
    ax.set_xticks(range(0, max_hour + 1, 2 if max_hour <= 48 else 6))

    # Cumulative reward / savings line (secondary axis)
    ax2 = ax.twinx()
    ax2.plot(
        df["hour"],
        df["cum_reward"],
        color=cum_reward_color,
        linewidth=2.0,
        linestyle="-.",
        label="Cum. reward (€)",
    )
    ax2.set_ylabel("Cumulative reward (€)", fontsize=11, fontweight="bold", color=cum_reward_color)
    ax2.tick_params(axis="y", labelcolor=cum_reward_color)
    ax2.grid(False)

    # Combine legends
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

    return fig, ax, ax2


def build_agent(
    agent_type: str,
    model_dir: str,
    model_name: str,
    battery_max_charge_kw: float,
    battery_max_discharge_kw: float,
    seed: int = 0,
) -> BaseAgent:
    """Instantiate the requested agent."""
    if agent_type == "sb3":
        return SB3Agent.load_or_train(
            model_dir=model_dir,
            model_name=model_name,
            seed=seed,
        )
    return RandomAgent(
        max_charge_kw=battery_max_charge_kw,
        max_discharge_kw=battery_max_discharge_kw,
        seed=seed,
    )


def run_simulation(
    num_days: int,
    agent_type: str,
    model_dir: str,
    model_name: str,
    steps_per_hour: int,
    battery_capacity_kwh: float,
    battery_max_charge_kw: float,
    battery_max_discharge_kw: float,
) -> pd.DataFrame:
    """Execute the simulation loop."""
    total_steps = num_days * 24 * steps_per_hour
    duration_hours = 1.0 / steps_per_hour

    battery = Battery(
        capacity_kwh=battery_capacity_kwh,
        max_charge_kw=battery_max_charge_kw,
        max_discharge_kw=battery_max_discharge_kw,
    )
    env = Environment(battery=battery, steps_per_hour=steps_per_hour, max_steps=total_steps)

    agent = build_agent(
        agent_type=agent_type,
        model_dir=model_dir,
        model_name=model_name,
        battery_max_charge_kw=battery_max_charge_kw,
        battery_max_discharge_kw=battery_max_discharge_kw,
    )

    records = []
    state = env.reset()
    cum_reward = 0.0

    for step in range(total_steps):
        state_dict = {
            "hour": state.hour,
            "battery_soc": state.battery_soc,
            "solar_yield": state.solar_yield,
            "demand_load": state.demand_load,
            "spot_price": state.spot_price,
        }

        # Select action
        action_kw = agent.act(state_dict)

        # Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action_kw)
        cum_reward += reward

        # Record metrics
        records.append(
            {
                "step": step,
                "hour": step * duration_hours,
                "solar_yield": state.solar_yield,
                "demand_load": state.demand_load,
                "spot_price": state.spot_price,
                "battery_soc": state.battery_soc,
                "action_kw": action_kw,
                "reward": reward,
                "cum_reward": cum_reward,
            }
        )

        state = next_state
        if terminated or truncated:
            break

    return pd.DataFrame(records)


if __name__ == "__main__":

    num_days: int = 3  # Number of days to simulate
    agent_type: str = "sb3"  # Option: "random" or "sb3"
    model_dir: str = "trained_agents"  # Directory where trained SB3 models are saved
    model_name: str = "sac_battery_policy"  # Model file stem (no ".zip" — SB3 appends it)
    steps_per_hour: int = 4  # Time resolution (e.g., 4 = 15 min steps)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"simulation_agent_{agent_type}.png"

    # Battery Parameters
    battery_capacity_kwh: float = 50.0
    battery_max_charge_kw: float = 20.0
    battery_max_discharge_kw: float = 20.0

    df = run_simulation(
        num_days=num_days,
        agent_type=agent_type,
        model_dir=model_dir,
        model_name=model_name,
        steps_per_hour=steps_per_hour,
        battery_capacity_kwh=battery_capacity_kwh,
        battery_max_charge_kw=battery_max_charge_kw,
        battery_max_discharge_kw=battery_max_discharge_kw,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")

    plot_exogenous_factors(df=df, fig=fig, ax=ax1)
    plot_agent_summary(df, ax=ax2)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
