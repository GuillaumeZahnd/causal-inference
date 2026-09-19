import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment import Environment, GridState


class GymEnvWrapper(gym.Env):
    """Gymnasium environment wrapping Environment with GridState objects."""

    def __init__(self, env_backend: Environment, max_kw: float = 20.0):
        super().__init__()
        self.backend = env_backend
        self.max_kw = max_kw

        # Continuous action space: [-max_discharge_kw, +max_charge_kw]
        self.action_space = spaces.Box(
            low=-self.max_kw, high=self.max_kw, shape=(1,), dtype=np.float32
        )

        # Observation space: [hour, battery_soc, solar_yield, demand_load, spot_price]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def _gridstate_to_array(self, state: GridState) -> np.ndarray:
        return np.array(
            [
                state.hour,
                state.battery_soc,
                state.solar_yield,
                state.demand_load,
                state.spot_price,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        grid_state = self.backend.reset()
        return self._gridstate_to_array(grid_state), {}

    def step(self, action: np.ndarray):
        action_scalar = float(action[0])
        next_grid_state, reward, terminated, truncated, info = self.backend.step(action_scalar)
        return (
            self._gridstate_to_array(next_grid_state),
            reward,
            terminated,
            truncated,
            info,
        )
