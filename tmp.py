import time
from pathlib import Path
import build._fused as fused
from build._fused import FusedSpheresCollisionChecker, SDFBase
from build._fused import Pose as Pose_
import tinyfk
import numpy as np
from skmp.robot.utils import load_collision_spheres
from skmp.robot.fetch import FetchConfig
from skrobot.model.primitives import Box
from skrobot.sdf import UnionSDF, BoxSDF, SphereSDF, CylinderSDF
np.random.seed(0)

def _sksdf_to_cppsdf(sksdf) -> SDFBase:
    if isinstance(sksdf, BoxSDF):
        pose = fused.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = fused.BoxSDF(sksdf._width, pose)
    elif isinstance(sksdf, CylinderSDF):
        pose = fused.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = fused.CylinderSDF(sksdf._radius, sksdf._height, pose)
    elif isinstance(sksdf, UnionSDF):
        for s in sksdf.sdf_list:
            if not isinstance(s, (BoxSDF, CylinderSDF)):
                raise ValueError("Unsupported SDF type")
        cpp_sdf_list = [_sksdf_to_cppsdf(s) for s in sksdf.sdf_list]
        sdf = fused.UnionSDF(cpp_sdf_list)
    else:
        raise ValueError("Unsupported SDF type")
    return sdf

urdf_model_path = tinyfk.fetch_urdfpath()
joint_names = [
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]


conf = FetchConfig()
obstacles = conf.get_self_body_obstacles()
table = Box([1.0, 2.0, 0.05], with_sdf=True)
table.translate([1.0, 0.0, 0.8])
ground = Box([2.0, 2.0, 0.05], with_sdf=True)
obstacles = [table] + obstacles + [ground]
sksdf = UnionSDF([o.sdf for o in obstacles])
cppsdf = _sksdf_to_cppsdf(sksdf)

coll_sphers = load_collision_spheres(Path("./fetch_coll_spheres.yaml"))
keys = list(coll_sphers.keys())
# revser order
# keys = keys[::-1]

parent_link_names = []
centers = []
radii = []
for key in keys:
    value = coll_sphers[key]
    for i in range(len(value)):
        parent_link_names.append(key)
        centers.append(value.center_list[i])
        radii.append(value.radius_list[i])

kin = FusedSpheresCollisionChecker(urdf_model_path, joint_names, parent_link_names, centers, radii, cppsdf)
start = np.array([ 0.,          1.31999949,  1.40000015, -0.20000077,  1.71999929,  0., 1.6600001,   0.        ])
goal = np.array([ 0.386,     0.20565826,  1.41370123,  0.30791941, -1.82230466,  0.24521043, 0.41718824,  6.01064401])
ret = kin.is_valid(start)
print(ret)
ret = kin.is_valid(goal)
print(ret)

# bench
N = 2000
Q = np.random.randn(N, 7)
ts = time.time()
for i in range(N):
    ret = kin.is_valid(Q[i])
print(f"per iter {(time.time() - ts) / N * 10 ** 6} [us]")

# solve rrt
from ompl import Algorithm, Planner
min_angles = np.array([0.0, -1.6056, -1.221, -np.pi * 2, -2.251, -np.pi * 2, -2.16, -np.pi * 2])
max_angles = np.array([0.38615, 1.6056, 1.518, np.pi * 2, 2.251, np.pi * 2, 2.16, np.pi * 2])
planner = Planner(min_angles, max_angles, lambda q: kin.is_valid(q), 10000, 0.1)

ts = time.time()
ret = planner.solve(start, goal, simplify=False)
print(f"planning time {1000 * (time.time() - ts)} [ms]")


from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import set_robot_state
from skrobot.models import Fetch
from skrobot.viewers import PyrenderViewer

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

self_body_obstacles = conf.get_self_body_obstacles()
for obs in self_body_obstacles:
    v.add(obs)
v.show()

time.sleep(1.0)
for q in ret:
    set_robot_state(fetch, conf.get_control_joint_names(), q)
    time.sleep(1.0)

import time; time.sleep(1000)
