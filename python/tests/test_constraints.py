import copy

import numpy as np
from fused.robot_spec import FetchSpec


def jac_numerical(const, q0: np.ndarray, eps: float) -> np.ndarray:
    f0, _ = const.evaluate(q0)
    dim_domain = len(q0)
    dim_codomain = len(f0)

    jac = np.zeros((dim_codomain, dim_domain))
    for i in range(dim_domain):
        q1 = copy.deepcopy(q0)
        q1[i] += eps
        f1, _ = const.evaluate(q1)
        jac[:, i] = (f1 - f0) / eps
    return jac


def check_jacobian(const, dim: int, eps: float = 1e-7, decimal: int = 4, std: float = 1.0):
    # check single jacobian
    for _ in range(10):
        q_test = np.random.randn(dim) * std
        _, jac_anal = const.evaluate(q_test)
        jac_numel = jac_numerical(const, q_test, eps)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)


def test_collision_free_constraint():
    fs = FetchSpec()
    fs.robot_model
    cst = fs.create_collision_const()
    for _ in range(10):
        q = np.random.randn(8)
        _, jac = cst.evaluate(q)
        jac_numel = jac_numerical(cst, q, 1e-6)
        np.testing.assert_almost_equal(jac, jac_numel, decimal=4)
