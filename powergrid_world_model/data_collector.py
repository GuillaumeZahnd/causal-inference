from agent import BaseAgent, RandomAgent
import pandas as pd
from battery import Battery
from environment import Environment


def collect_simulation_data(
    agent: BaseAgent,
    nb_days: int ,
    steps_per_hour: int,
    seed: int,
    ) -> pd.DataFrame:

    """Steps through the simulation and logs (S_t, A_t, R_t, S_{t+1}) trajectories."""

    battery = Battery()
    env = Environment(battery=battery, seed=seed)
    duration_hours = 1.0 / steps_per_hour
    total_steps = nb_days * 24 * steps_per_hour

    records = []

    next_grid_state = env.get_state(0)

    for step_idx in range(total_steps):

        # Fetch current state (GridState instance)
        grid_state = next_grid_state

        # Agent selects action (positive sign: we import, negative sign: we export)
        action_kw = agent.act(grid_state.to_dict())

        # Step the physical battery
        delta_soc_kwh = battery.step(
            action_kw=action_kw,
            duration_hours=duration_hours
        )

        # Net power flowing from us to the grid (positive: net import, negative: net export)
        net_grid_kw = grid_state.demand_load - grid_state.solar_yield + action_kw

        # Financial cost (positive: we pay, negative: we get paid)
        grid_cost = net_grid_kw * grid_state.spot_price * duration_hours

        # Convert the financial cost to rewards (ideally we "buy low, sell high")
        reward = -grid_cost

        # Peek at the next exogenous state
        next_grid_state = env.get_state(step_idx + 1)

        # Log trajectory record
        records.append({
            "step_idx": step_idx,
            "hour": grid_state.hour,
            "battery_soc": grid_state.battery_soc,  # Current state (S_t)
            "solar_yield": grid_state.solar_yield,
            "demand_load": grid_state.demand_load,
            "spot_price": grid_state.spot_price,
            "next_hour": next_grid_state.hour,
            "next_battery_soc": next_grid_state.battery_soc,  # Next state (S_{t+1})
            "next_solar_yield": next_grid_state.solar_yield,
            "next_demand_load": next_grid_state.demand_load,
            "next_spot_price": next_grid_state.spot_price,
            "action_kw": action_kw,  # Treatment (A_t)
            "delta_soc_kwh": delta_soc_kwh,  # Transition outcome
            "reward": reward,  # Outcome (Y_t)
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    agent = RandomAgent(seed=42)

    df = collect_simulation_data(
        agent=agent,
        nb_days=60,
        steps_per_hour=4,
        seed=0
    )

    print(f"Logged dataset shape: {df.shape}")
    print(df.head())
