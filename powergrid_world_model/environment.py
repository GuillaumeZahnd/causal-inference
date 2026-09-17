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
    solar_yield: float
    demand_load: float
    spot_price: float


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
    """Generates exogenous environment variables for a single step.

    Args:
        step_idx: Discrete time step index.
        steps_per_hour: Granularity of time steps (1 = hourly, 4 = 15-min
          intervals).
        rng: Optional NumPy Random Generator for stochastic reproducibility.

    Returns:
        Dict containing hour, solar_yield, demand_load, and spot_price.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total_steps_per_day = 24 * steps_per_hour
    hour = (step_idx % total_steps_per_day) / steps_per_hour

    # Diurnal solar yield (peaks around noon)
    solar_base = max(0.0, np.sin(np.pi * (hour - 6) / 12))
    solar_yield = float(
        max(0.0, solar_base * 10.0 * rng.uniform(0.85, 1.0))
        )

    # Dual-peak demand load curve (morning @ 08:00, evening @ 19:00)
    morning_peak = np.exp(-(((hour - 8) / 1.5) ** 2))
    evening_peak = np.exp(-(((hour - 19) / 2.0) ** 2))
    base_load = 1.5 + rng.normal(0, 0.1)
    demand_load = float(max(0.2, base_load + (3.5 * morning_peak) + (5.0 * evening_peak)))

    # Spot price ($/kWh): Increases under high demand relative to solar
    spot_price = float(max(0.01, 0.10 + 0.05 * demand_load - 0.02 * solar_yield))

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
        ):

        self.rng = np.random.default_rng(seed)
        self.steps_per_hour = steps_per_hour
        self.battery = battery

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
    seed: int = 42
    ) -> pd.DataFrame:

    """Generate a complete multi-step dataset for analysis and plotting."""
    rng = np.random.default_rng(seed)
    total_steps = 24 * steps_per_hour

    records = [
        generate_environment_step(step_idx=i, steps_per_hour=steps_per_hour, rng=rng)
        for i in range(total_steps)
        ]

    return pd.DataFrame(records)
