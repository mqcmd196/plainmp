import time

from fused.ik import solve_ik
from fused.robot_spec import JaxonSpec
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import set_robot_state
from skrobot.viewers import PyrenderViewer

import tinyfk

jspec = JaxonSpec()
com_const = jspec.create_default_com_const()
stand_pose_const = jspec.create_default_stand_pose_const()
lb, ub = jspec.angle_bounds()
ret = solve_ik(stand_pose_const, None, lb, ub, None)
ret = solve_ik(stand_pose_const, com_const, lb, ub, ret.q)

# visualize
v = PyrenderViewer()
robot = Jaxon()
set_robot_state(robot, jspec.control_joint_names, ret.q, tinyfk.BaseType.FLOATING)
v.add(robot)
v.show()
time.sleep(10)
