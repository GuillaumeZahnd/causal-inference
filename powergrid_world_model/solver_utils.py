from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from battery import Battery
from environment import Environment


@dataclass
class SolverConfig:
    battery: Battery = field(default_factory=Battery)
    soc_grid_points: int = 201    # Resolution of SOC discretization
    action_grid_points: int = 41  # Resolution of action discretization

    @property
    def capacity_kwh(self) -> float:
        return self.battery.capacity_kwh

    @property
    def max_charge_kw(self) -> float:
        return self.battery.max_charge_kw

    @property
    def max_discharge_kw(self) -> float:
        return self.battery.max_discharge_kw

    @property
    def efficiency(self) -> float:
        return self.battery.efficiency


def compute_grid_reward(
    demand_load: float,
    solar_yield: float,
    delta_grid_kw: float,
    spot_price: float,
    duration_hours: float,
) -> float:

    net_grid_kw = demand_load - solar_yield + delta_grid_kw
    grid_cost = net_grid_kw * spot_price * duration_hours

    return -grid_cost


def generate_day_trajectory(
    seed: int,
    steps_per_hour: int
) -> pd.DataFrame:

    battery = Battery()  # Unused except to satisfy Environment's constructor
    env = Environment(battery=battery, seed=seed, steps_per_hour=steps_per_hour)
    total_steps = 24 * steps_per_hour

    records = []
    for step_idx in range(total_steps):
        state = env.get_state(step_idx)
        records.append({
            "step_idx": int(step_idx),
            "hour": int(state.hour),
            "solar_yield": float(state.solar_yield),
            "demand_load": float(state.demand_load),
            "spot_price": float(state.spot_price),
        })
    return pd.DataFrame(records)


def battery_transition(
    soc: float,
    action_kw: float,
    duration_hours: float,
    cfg: SolverConfig,
) -> tuple[float, float]:

    delta_soc_kwh, delta_grid_kw = Battery.compute_step(
        soc_kwh=soc,
        action_kw=action_kw,
        duration_hours=duration_hours,
        capacity_kwh=cfg.capacity_kwh,
        max_charge_kw=cfg.max_charge_kw,
        max_discharge_kw=cfg.max_discharge_kw,
        efficiency=cfg.efficiency,
    )

    next_soc = float(np.clip(soc + delta_soc_kwh, 0.0, cfg.capacity_kwh))

    return next_soc, delta_grid_kw


def step_reward(
    soc: float,
    action_kw: float,
    solar_yield: float,
    demand_load: float,
    spot_price: float,
    duration_hours: float,
    cfg: SolverConfig,
) -> tuple[float, float, float]:

    next_soc, delta_grid_kw = battery_transition(soc, action_kw, duration_hours, cfg)

    reward = compute_grid_reward(
        demand_load=demand_load,
        solar_yield=solar_yield,
        delta_grid_kw=delta_grid_kw,
        spot_price=spot_price,
        duration_hours=duration_hours,
    )

    return reward, next_soc, delta_grid_kw
