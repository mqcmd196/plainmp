from typing import Sequence, Tuple

import numpy as np
from fused.psdf import PrimitiveSDFBase

class ConstraintBase:
    def update_kintree(self, q: np.ndarray) -> None: ...
    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]: ...

class EqConstraintBase(ConstraintBase): ...

class IneqConstraintBase(ConstraintBase):
    def is_valid(self) -> bool: ...

class LinkPoseCst(EqConstraintBase): ...

class SphereCollisionCst(IneqConstraintBase):
    def set_sdfs(self, sdfs: Sequence[PrimitiveSDFBase]) -> None: ...
