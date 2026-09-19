from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3 import SAC


class BaseAgent(ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def act(self, state: dict) -> float:
        """
        Given a state dict S_t, return an action A_t.
        The action corresponds to the requested power flow, in kW (+ for charge, - for discharge, 0 for hold).
        """
        pass


class RandomAgent(BaseAgent):
    """Exploratory agent providing uniform coverage over the continuous action space."""

    def __init__(
        self,
        max_charge_kw: float = 20.0,
        max_discharge_kw: float = 20.0,
        seed: int = 0,
    ):
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.rng = np.random.default_rng(seed)

    def act(self, state: dict) -> float:
        """Sample action uniformly from [-max_discharge, +max_charge]."""
        return float(self.rng.uniform(-self.max_discharge_kw, self.max_charge_kw))


class SB3Agent(BaseAgent):
    """Agent driven by a trained Stable-Baselines3 model (e.g. SAC or PPO)."""

    def __init__(self, model_path: str | None = None, model: SAC | None = None):
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = SAC.load(model_path)
        else:
            raise ValueError("Must provide either a trained model instance or a model_path.")

    def _dict_to_array(self, state: dict) -> np.ndarray:
        return np.array(
            [
                state["hour"],
                state["battery_soc"],
                state["solar_yield"],
                state["demand_load"],
                state["spot_price"],
            ],
            dtype=np.float32,
        )

    def act(self, state: dict) -> float:
        obs = self._dict_to_array(state)
        # deterministic=False retains entropy/exploration if collecting diverse WM data
        action, _ = self.model.predict(obs, deterministic=True)
        return float(action[0])
