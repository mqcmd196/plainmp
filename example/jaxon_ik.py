import time

from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import set_robot_state
from skrobot.viewers import PyrenderViewer

import tinyfk
from plainmp.constraint import EqCompositeCst
from plainmp.ik import solve_ik
from plainmp.robot_spec import JaxonSpec

jspec = JaxonSpec()
com_const = jspec.create_default_com_const()
stand_pose_const = jspec.create_default_stand_pose_const()
rarm_pose_const = jspec.create_pose_const(["rarm_end_coords"], [[0.7, -0.6, 0.0]])
eq_cst = EqCompositeCst([rarm_pose_const, stand_pose_const])
lb, ub = jspec.angle_bounds()
ret = solve_ik(eq_cst, None, lb, ub, None)
ret = solve_ik(eq_cst, com_const, lb, ub, ret.q)
assert ret.success

# visualize
v = PyrenderViewer()
robot = Jaxon()
set_robot_state(robot, jspec.control_joint_names, ret.q, tinyfk.BaseType.FLOATING)
v.add(robot)
v.show()
time.sleep(10)
