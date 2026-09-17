from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for grid decision agents."""

    @abstractmethod
    def act(self, state: dict) -> float:
        """Given a state dict S_t, return an action A_t in kW (+ for charge, - for discharge, 0 for hold)."""
        pass


class RandomAgent(BaseAgent):
    """Exploratory agent providing uniform coverage over the continuous action space."""

    def __init__(
        self,
        max_charge_kw: float = 20.0,
        max_discharge_kw: float = 20.0,
        seed: int = 42,
    ):
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.rng = np.random.default_rng(seed)

    def act(self, _state: dict) -> float:
        """Sample action uniformly from [-max_discharge, +max_charge]."""
        return float(self.rng.uniform(-self.max_discharge_kw, self.max_charge_kw))


class RuleBasedAgent(BaseAgent):
    """Heuristic agent: charges when solar > demand, discharges during high spot prices."""

    def __init__(
        self,
        max_charge_kw: float = 20.0,
        max_discharge_kw: float = 20.0,
        high_price_threshold: float = 0.25,
    ):
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.high_price_threshold = high_price_threshold

    def act(self, state: dict) -> float:
        net_power = state["solar_yield"] - state["demand_load"]

        # Excess solar -> charge battery
        if net_power > 0:
            return float(min(net_power, self.max_charge_kw))

        # Peak spot price -> discharge to offset grid cost
        if state["spot_price"] >= self.high_price_threshold:
            return float(-self.max_discharge_kw)

        # Default hold
        return 0.0
