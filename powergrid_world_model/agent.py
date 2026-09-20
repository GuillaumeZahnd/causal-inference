from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3 import SAC
from pathlib import Path
import hydra


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

    def __init__(
        self,
        model_path: str | None = None,
        model: SAC | None = None
    ):
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

    @classmethod
    def load_or_train(
        cls,
        model_dir: str,
        model_name: str,
        total_timesteps: int = 20_000,
        seed: int = 0,
    ) -> "SB3Agent":
        """Factory method to load an existing SB3 model or train one if missing."""
        from battery import Battery
        from environment import Environment
        from gym_env_wrapper import GymEnvWrapper

        dir_path = Path(model_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        try:
            base_dir = Path(hydra.utils.get_original_cwd())
        except (ValueError, RuntimeError):
            base_dir = Path.cwd()

        dir_path = base_dir / model_dir
        dir_path.mkdir(parents=True, exist_ok=True)

        model_path = dir_path / model_name
        zip_path = model_path.with_suffix(".zip")

        if not zip_path.exists():
            print(f"No existing model found at {zip_path}. Training new SB3 agent...")
            gym_env = GymEnvWrapper(env_backend=Environment(battery=Battery(), seed=seed))

            # Correctly reference variable 'model' across all calls
            model = SAC("MlpPolicy", gym_env, verbose=1, learning_rate=3e-4, seed=seed)
            model.learn(total_timesteps=total_timesteps)
            model.save(str(model_path))
            print(f"Model saved to {zip_path}")

        return cls(model_path=str(model_path))
