from typing import Tuple

import numpy as np

class LinkPoseCst:
    def evaluate(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

class SphereCollisionCst:
    def evaluate(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
