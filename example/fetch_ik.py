import time

from plainmp.ik import solve_ik
from plainmp.robot_spec import FetchSpec
from plainmp.utils import sksdf_to_cppsdf
from skmp.robot.utils import set_robot_state
from skrobot.model.primitives import Box
from skrobot.models.fetch import Fetch
from skrobot.viewers import PyrenderViewer

# create table sdf
table = Box([1.0, 2.0, 0.05], with_sdf=True)
table.translate([1.0, 0.0, 0.8])
sdf = sksdf_to_cppsdf(table.sdf)

# create problem
fs = FetchSpec()
eq_cst = fs.create_gripper_pose_const([0.7, +0.2, 0.95, 0, 0, 0])
ineq_cst = fs.create_collision_const()
ineq_cst.set_sdfs([sdf])

lb, ub = fs.angle_bounds()
ts = time.time()

for i in range(10):
    ret = solve_ik(eq_cst, ineq_cst, lb, ub, None)
    if ret.success:
        break
# in msec
print(f"after {i} trials, elapsed time: {(time.time() - ts) * 1000:.2f} msec")
assert ret.success

# visualization
fetch = Fetch()
set_robot_state(fetch, fs.control_joint_names, ret.q)
v = PyrenderViewer()
v.add(fetch)
v.add(table)
v.show()
time.sleep(1000)
