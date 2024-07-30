import yaml
from typing import Any, Optional, TypeVar, Generic, List, Tuple, Dict, Type
import time
from pathlib import Path
import build._fused as fused
from build._fused import FusedSpheresCollisionChecker, SphereAttachentSpec
import build._fused.primitive_sdf as psdf 
import tinyfk
import numpy as np
from skmp.robot.utils import load_collision_spheres
from skmp.robot.fetch import FetchConfig
from skrobot.model.primitives import Box, Cylinder
from skrobot.sdf import UnionSDF, BoxSDF, SphereSDF, CylinderSDF

np.random.seed(0)
def _sksdf_to_cppsdf(sksdf) -> psdf.SDFBase:
    if isinstance(sksdf, BoxSDF):
        pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = psdf.BoxSDF(sksdf._width, pose)
    elif isinstance(sksdf, CylinderSDF):
        pose = psdf.Pose(sksdf.worldpos(), sksdf.worldrot())
        sdf = psdf.CylinderSDF(sksdf._radius, sksdf._height, pose)
    elif isinstance(sksdf, UnionSDF):
        for s in sksdf.sdf_list:
            if not isinstance(s, (BoxSDF, CylinderSDF)):
                raise ValueError("Unsupported SDF type")
        cpp_sdf_list = [_sksdf_to_cppsdf(s) for s in sksdf.sdf_list]
        sdf = psdf.UnionSDF(cpp_sdf_list)
    else:
        raise ValueError("Unsupported SDF type")
    return sdf


def load_collision_spheres(yaml_file_path: Path) -> List[SphereAttachentSpec]:
    with open(yaml_file_path, "r") as f:
        collision_config = yaml.safe_load(f)
    d = collision_config["collision_spheres"]

    def unique_name(link_name) -> str:
        return link_name + str(uuid.uuid4())[:13]

    specs = []
    for link_name, vals in d.items():
        ignore_collision = vals["ignore_collision"]
        spheres_d = vals["spheres"]
        for spec in spheres_d:
            vals = np.array(spec)
            center, r = vals[:3], vals[3]
            specs.append(SphereAttachentSpec(link_name, center, r, ignore_collision))
    return specs

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
sdfs = [_sksdf_to_cppsdf(o.sdf) for o in obstacles]

specs = load_collision_spheres(Path("./fetch_coll_spheres.yaml"))

kin = FusedSpheresCollisionChecker(urdf_model_path, joint_names, specs, [], sdfs)
start = np.array([ 0.,          1.31999949,  1.40000015, -0.20000077,  1.71999929,  0., 1.6600001,   0.        ])
goal = np.array([ 0.386,     0.20565826,  1.41370123,  0.30791941, -1.82230466,  0.24521043, 0.41718824,  6.01064401])

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
    colvis.update(fetch)
    time.sleep(1.0)

import time; time.sleep(1000)
