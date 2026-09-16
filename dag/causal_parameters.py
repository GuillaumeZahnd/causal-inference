from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class CausalParameters:
    cate: np.ndarray
    cate_lower: np.ndarray | None = None
    cate_upper: np.ndarray | None = None
    ate: float | None = None
    ate_lower: float | None = None
    ate_upper: float | None = None

    def __post_init__(self) -> None:
        if self.ate is None:
            self.ate = float(np.mean(self.cate))
