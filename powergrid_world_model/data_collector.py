from agent import BaseAgent, RandomAgent
import pandas as pd
from battery import Battery
from environment import Environment


def collect_simulation_data(
    agent: BaseAgent,
    num_days: int = 30,
    steps_per_hour: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Steps through the simulation and logs (S_t, A_t, R_t, S_{t+1}) trajectories."""

    battery = Battery()
    env = Environment(battery=battery, seed=seed)
    duration_hours = 1.0 / steps_per_hour
    total_steps = num_days * 24 * steps_per_hour

    records = []

    for step_idx in range(total_steps):

        # Fetch current state (GridState instance)
        grid_state = env.get_state(step_idx)

        # Agent selects action
        state_dict = grid_state.to_dict()  #
        action_kw = agent.act(state_dict)

        # Step the physical battery
        delta_soc_kwh = battery.step(
            action_kw=action_kw, duration_hours=duration_hours
        )

        # Financial Cost / Reward calculation using attribute access
        net_grid_kw = (grid_state.demand_load - grid_state.solar_yield + action_kw)
        grid_cost = net_grid_kw * grid_state.spot_price * duration_hours
        reward = -grid_cost

        # Log trajectory record
        records.append({
            "step_idx": step_idx,
            "hour": grid_state.hour,
            "battery_soc": grid_state.battery_soc,
            "solar_yield": grid_state.solar_yield,
            "demand_load": grid_state.demand_load,
            "spot_price": grid_state.spot_price,
            "action_kw": action_kw,  # Treatment (A_t)
            "delta_soc_kwh": delta_soc_kwh,  # Transition Outcome
            "reward": reward,  # Outcome (Y_t)
            "next_battery_soc": battery.current_soc_kwh,  # Next State (S_{t+1})
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    agent = RandomAgent(seed=42)
    df_dataset = collect_simulation_data(agent=agent, num_days=60)
    print(f"Logged dataset shape: {df_dataset.shape}")
    print(df_dataset.head())
