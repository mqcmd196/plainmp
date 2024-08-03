from dataclasses import dataclass
from typing import Optional, Union

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
