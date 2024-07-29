import time
from pathlib import Path
import build._fused as fused
from build._fused import FusedSpheresCollisionChecker, SDFBase, SphereAttachentSpec, SDFAttachmentSpec
import tinyfk
import numpy as np
from skmp.robot.utils import load_collision_spheres
from skmp.robot.fetch import FetchConfig
from skrobot.model.primitives import Box, Cylinder
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

sphere_spec_list = []
for key in keys:
    value = coll_sphers[key]
    for i in range(len(value)):
        spec = SphereAttachentSpec(key, value.center_list[i], value.radius_list[i])
        sphere_spec_list.append(spec)


# attach sdfs
neck_lower = Box([0.1, 0.18, 0.08], face_colors=[255, 255, 255, 200], with_sdf=True)
neck_lower.translate([0.0, 0.0, 0.97])
neck_upper = Box([0.05, 0.17, 0.15], face_colors=[255, 255, 255, 200], with_sdf=True)
neck_upper.translate([-0.035, 0.0, 0.92])
head = Cylinder(0.235, 0.12, face_colors=[255, 255, 255, 200], with_sdf=True)
head.translate([0.0, 0.0, 1.04])
torso_position =np.array([-0.086875, 0., 0.37743 ])
neck_lower.translate(-torso_position)
neck_upper.translate(-torso_position)
head.translate(-torso_position)

sdf_specs = []
for o in [neck_lower, neck_upper, head]:
    primitive_sdf = _sksdf_to_cppsdf(o.sdf)
    print(primitive_sdf)
    sdf_attachment_spec = SDFAttachmentSpec("torso_lift_link", o.worldpos(), primitive_sdf)
    sdf_specs.append(sdf_attachment_spec)


kin = FusedSpheresCollisionChecker(urdf_model_path, joint_names, sphere_spec_list, sdf_specs, cppsdf)
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
ret = planner.solve(start, goal, simplify=True)
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
