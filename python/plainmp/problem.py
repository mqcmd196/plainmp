from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np

from plainmp.constraint import EqConstraintBase, IneqConstraintBase


@dataclass
class Problem:
    start: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    goal_const: Union[EqConstraintBase, np.ndarray]
    global_ineq_const: Optional[IneqConstraintBase]
    global_eq_const: Optional[EqConstraintBase]
    motion_step_box: np.ndarray

    def check_init_feasibility(self) -> Tuple[bool, str]:
        if not (np.all(self.lb <= self.start) and np.all(self.start <= self.ub)):
            return False, "Start point is out of bounds"
        if self.global_ineq_const is not None:
            if not self.global_ineq_const.is_valid(self.start):
                return False, "Start point violates global inequality constraints"
        return True, ""
