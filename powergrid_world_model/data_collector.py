from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
from stable_baselines3 import SAC

from agent import BaseAgent, RandomAgent, SB3Agent
from battery import Battery
from environment import Environment
from gym_env_wrapper import GymEnvWrapper
from solver_utils import compute_grid_reward


def collect_simulation_data(
    agent: BaseAgent,
    nb_days: int,
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
        # Fetch current state
        grid_state = next_grid_state

        # Agent selects action
        action_kw = agent.act(grid_state.to_dict())

        # Step physical battery stateful object
        delta_soc_kwh, delta_grid_kw = battery.step(
            action_kw=action_kw,
            duration_hours=duration_hours,
        )

        # Compute rewards
        reward = compute_grid_reward(
            demand_load=float(grid_state.demand_load),
            solar_yield=float(grid_state.solar_yield),
            delta_grid_kw=delta_grid_kw,
            spot_price=float(grid_state.spot_price),
            duration_hours=duration_hours,
        )

        # Peek at the next exogenous state
        next_grid_state = env.get_state(step_idx + 1)

        # Log trajectory record
        records.append({
            "step_idx": int(step_idx),
            "hour": int(grid_state.hour),
            "battery_soc": float(grid_state.battery_soc),
            "solar_yield": float(grid_state.solar_yield),
            "demand_load": float(grid_state.demand_load),
            "spot_price": float(grid_state.spot_price),
            "next_hour": int(next_grid_state.hour),
            "next_battery_soc": float(next_grid_state.battery_soc),
            "next_solar_yield": float(next_grid_state.solar_yield),
            "next_demand_load": float(next_grid_state.demand_load),
            "next_spot_price": float(next_grid_state.spot_price),
            "action_kw": float(action_kw),
            "delta_soc_kwh": float(delta_soc_kwh),
            "delta_grid_kw": float(delta_grid_kw),
            "reward": float(reward),
        })

    return pd.DataFrame(records)


def get_agent(cfg: DictConfig) -> BaseAgent:
    agent_type = cfg.agent.type.lower()

    if agent_type == "random":
        return RandomAgent(seed=cfg.simulation.seed)

    elif agent_type == "sb3":
        model_dir = Path(hydra.utils.get_original_cwd()) / cfg.sb3.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / cfg.sb3.model_name
        zip_path = model_path.with_suffix(".zip")

        if not zip_path.exists():
            print(f"No existing model found at {zip_path}. Training new SB3 agent...")
            gym_env = GymEnvWrapper(env_backend=Environment(battery=Battery()))
            model = SAC("MlpPolicy", gym_env, verbose=1, learning_rate=3e-4)
            model.learn(total_timesteps=cfg.sb3.total_timesteps)
            model.save(str(model_path))
            print(f"Model saved to {zip_path}")
        else:
            print(f"Loading existing SB3 agent model from {zip_path}")

        return SB3Agent(model_path=str(model_path))

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    agent = get_agent(cfg)

    df = collect_simulation_data(
        agent=agent,
        nb_days=cfg.simulation.nb_days,
        steps_per_hour=cfg.simulation.steps_per_hour,
        seed=cfg.simulation.seed,
    )

    print(f"Logged dataset shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
