from dataclasses import dataclass
import numpy as np
import pandas as pd

from battery import Battery


@dataclass(frozen=True)
class GridState:
    """State vector (S_t) passed to the Causal World Model."""

    step_idx: int
    hour: float
    battery_soc: float
    solar_yield: float  # How much power the solar panels are currently producing ("free supply")
    demand_load: float  # How much power the site is currently consuming ("need")
    spot_price: float

    """
    demand_load - solar_yield: baseline gap to cover from the grid
        positive means we are short (need to import), negative means we have surplus (could export)
    """

    def to_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "hour": self.hour,
            "battery_soc": self.battery_soc,
            "solar_yield": self.solar_yield,
            "demand_load": self.demand_load,
            "spot_price": self.spot_price,
            }


def generate_environment_step(
    step_idx: int,
    steps_per_hour: int = 1,
    rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
    """Generate exogenous environment variables for a single step.

    Args:
        step_idx: Discrete time step index.
        steps_per_hour: Granularity of time steps (1 = hourly, 4 = 15-min intervals).
        rng: Optional NumPy Random Generator for stochastic reproducibility.

    Returns:
        Dictionary containing hour, solar_yield, demand_load, and spot_price.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total_steps_per_day = 24 * steps_per_hour
    hour = (step_idx % total_steps_per_day) / steps_per_hour

    # Diurnal solar yield (peaks around noon)
    solar_base = max(0.0, np.sin(np.pi * (hour - 6) / 12))
    solar_yield = float(max(0.0, solar_base * 30.0 * rng.uniform(0.75, 1.0)))

    # Dual-peak demand load curve (morning @ 08:00, evening @ 19:00)
    morning_peak = np.exp(-(((hour - 8) / 1.5) ** 2))
    evening_peak = np.exp(-(((hour - 19) / 2.0) ** 2))
    base_load = 1.5 + rng.normal(0, 0.6)
    demand_load = float(max(0.2, base_load + (3.5 * morning_peak) + (5.0 * evening_peak)))

    # Spot price (€/kWh): rises with demand, falls with solar supply.
    # Allowed to go mildly negative during high-solar/low-demand periods,
    # mirroring real "duck curve" markets where excess renewable supply can push wholesale prices below zero.
    spot_price = float(0.10 + 0.05 * demand_load - 0.02 * solar_yield)
    spot_price = max(-0.05, spot_price)  # floor at a small negative value, not zero

    return {
        "hour": hour,
        "solar_yield": solar_yield,
        "demand_load": demand_load,
        "spot_price": spot_price,
        }


class Environment:
    """Class wrapper for sequential environment progression."""

    def __init__(
        self,
        battery: Battery,
        seed: int = 42,
        steps_per_hour: int = 1,
        max_steps: int = 24,  # Define episode length (e.g., 24 hours)
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.steps_per_hour = steps_per_hour
        self.battery = battery
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self) -> GridState:
        """Reset battery state and step counter for a new episode."""
        self.current_step = 0
        self.rng = np.random.default_rng(self.seed)
        if hasattr(self.battery, "reset"):
            self.battery.reset()
        return self.get_state(step_idx=self.current_step)

    def step(self, action_kw: float) -> tuple[GridState, float, bool, bool, dict]:
        """Advance environment by one step given a battery control action."""
        # Get state before action
        state = self.get_state(step_idx=self.current_step)

        # Calculate step duration in hours from steps_per_hour
        duration_hours = 1.0 / self.steps_per_hour

        # Apply action with duration_hours argument
        actual_power, _ = self.battery.step(action_kw, duration_hours=duration_hours)

        # Calculate step net energy cost / reward
        net_grid_kw = state.demand_load - state.solar_yield + actual_power
        cost = net_grid_kw * state.spot_price * duration_hours
        reward = -cost  # Maximizing reward

        # Advance step counter
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        next_state = self.get_state(step_idx=self.current_step)

        return next_state, reward, terminated, truncated, {}

    def get_state(self, step_idx: int) -> GridState:
        exogenous_state = generate_environment_step(
            step_idx=step_idx,
            steps_per_hour=self.steps_per_hour,
            rng=self.rng,
        )
        return GridState(
            step_idx=step_idx,
            hour=exogenous_state["hour"],
            battery_soc=self.battery.current_soc_kwh,
            solar_yield=exogenous_state["solar_yield"],
            demand_load=exogenous_state["demand_load"],
            spot_price=exogenous_state["spot_price"],
        )


def generate_synthetic_day_data(
    steps_per_hour: int = 4,
    seed: int = 0
    ) -> pd.DataFrame:

    """Generate a complete multi-step dataset for analysis and plotting."""
    rng = np.random.default_rng(seed)
    total_steps = 24 * steps_per_hour

    records = [
        generate_environment_step(step_idx=i, steps_per_hour=steps_per_hour, rng=rng)
        for i in range(total_steps)
        ]

    return pd.DataFrame(records)
