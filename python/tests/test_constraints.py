import copy

import numpy as np
import pytest
from fused.constraint import AppliedForceSpec, ComInPolytopeCst
from fused.psdf import BoxSDF, Pose
from fused.robot_spec import FetchSpec


def jac_numerical(const, q0: np.ndarray, eps: float) -> np.ndarray:
    const.update_kintree(q0)
    f0, _ = const.evaluate()
    dim_domain = len(q0)
    dim_codomain = len(f0)

    jac = np.zeros((dim_codomain, dim_domain))
    for i in range(dim_domain):
        q1 = copy.deepcopy(q0)
        q1[i] += eps
        const.update_kintree(q1)
        f1, _ = const.evaluate()
        jac[:, i] = (f1 - f0) / eps
    return jac


def check_jacobian(const, dim: int, eps: float = 1e-7, decimal: int = 4, std: float = 1.0):
    # check single jacobian
    for _ in range(10):
        q_test = np.random.randn(dim) * std
        const.update_kintree(q_test)
        _, jac_anal = const.evaluate()
        jac_numel = jac_numerical(const, q_test, eps)
        np.testing.assert_almost_equal(jac_anal, jac_numel, decimal=decimal)


@pytest.mark.parametrize("with_rpy", [False, True])
def test_link_pose_constraint(with_rpy: bool):
    if with_rpy:
        pose = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    else:
        pose = [0.7, 0.0, 0.7]

    fs = FetchSpec()
    cst = fs.create_gripper_pose_const(pose)
    check_jacobian(cst, 8)


def test_link_pose_constraint_multi_link():
    pose1 = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
    pose2 = [0.7, 0.0, 0.7]
    fs = FetchSpec()
    cst = fs.create_pose_const(["gripper_link", "wrist_roll_link"], [pose1, pose2])
    check_jacobian(cst, 8)


def test_collision_free_constraint():
    fs = FetchSpec()
    sdf = BoxSDF([1, 1, 1], Pose([0.5, 0.5, 0.5], np.eye(3)))
    for self_collision in [False, True]:
        cst = fs.create_collision_const(self_collision)
        cst.set_sdfs([sdf])
        check_jacobian(cst, 8)


@pytest.mark.parametrize("with_force", [False, True])
def test_com_in_polytope_constraint(with_force: bool):
    fs = FetchSpec()
    sdf = BoxSDF([0.3, 0.3, 0], Pose([0.0, 0.0, 0.0], np.eye(3)))
    afspecs = []
    if with_force:
        afspecs.append(AppliedForceSpec("gripper_link", 2.0))
    cst = ComInPolytopeCst(fs.get_kin(), fs.control_joint_names, sdf, afspecs)
    check_jacobian(cst, 8)


if __name__ == "__main__":
    test_com_in_polytope_constraint()
