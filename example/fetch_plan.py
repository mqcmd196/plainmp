import time

import numpy as np
from fused.robot_spec import FetchSpec
from fused.utils import sksdf_to_cppsdf
from ompl import Planner
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import set_robot_state
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager
from skrobot.model.primitives import Box
from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

fetch = FetchSpec()
cst = fetch.create_collision_const()

table = Box([1.0, 2.0, 0.05], with_sdf=True)
table.translate([1.0, 0.0, 0.8])
ground = Box([2.0, 2.0, 0.05], with_sdf=True)
sdfs = [sksdf_to_cppsdf(table.sdf), sksdf_to_cppsdf(ground.sdf)]
cst.set_sdfs(sdfs)

min_angles = np.array([0.0, -1.6056, -1.221, -np.pi * 2, -2.251, -np.pi * 2, -2.16, -np.pi * 2])
max_angles = np.array([0.38615, 1.6056, 1.518, np.pi * 2, 2.251, np.pi * 2, 2.16, np.pi * 2])
planner = Planner(
    min_angles,
    max_angles,
    lambda q: cst.is_valid(q),
    10000,
    [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2],
)
start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
goal = np.array([0.386, 0.20565, 1.41370, 0.30791, -1.82230, 0.24521, 0.41718, 6.01064])
times = []
for _ in range(100):
    ts = time.time()
    ret = planner.solve(start, goal, simplify=False)
    print(f"planning time {1000 * (time.time() - ts)} [ms]")
    times.append(time.time() - ts)
print(f"average planning time {1000 * np.mean(times)} [ms]")

# visualize
conf = FetchConfig()
fetch = Fetch()
set_robot_state(fetch, conf.get_control_joint_names(), goal)
v = PyrenderViewer()
colkin = conf.get_collision_kin()
colvis = CollisionSphereVisualizationManager(colkin, v)
colvis.update(fetch)
v.add(fetch)
v.add(table)
v.add(ground)
v.show()

time.sleep(1.0)
for q in ret:
    set_robot_state(fetch, conf.get_control_joint_names(), q)
    colvis.update(fetch)
    v.redraw()
    time.sleep(0.3)

import time

time.sleep(1000)
