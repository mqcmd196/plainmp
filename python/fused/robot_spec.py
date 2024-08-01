import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import yaml
from fused.constraint import LinkPoseCst, SphereAttachentSpec, SphereCollisionCst
from fused.tinyfk import KinematicModel
from fused.utils import sksdf_to_cppsdf
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rotation_matrix, rpy_angle
from skrobot.model.primitives import Box, Cylinder, Sphere
from skrobot.model.robot_model import RobotModel
from skrobot.utils.urdf import URDF, no_mesh_load_mode

_loaded_urdf_models: Dict[str, URDF] = {}


def load_urdf_model_using_cache(file_path: Path, deepcopy: bool = True):
    file_path = file_path.expanduser()
    assert file_path.exists()
    key = str(file_path)
    if key not in _loaded_urdf_models:
        with no_mesh_load_mode():
            urdf = URDF.load(str(file_path))
        _loaded_urdf_models[key] = urdf
    if deepcopy:
        return copy.deepcopy(_loaded_urdf_models[key])
    else:
        return _loaded_urdf_models[key]


class RobotSpec(ABC):
    def __init__(self, conf_file: Path, with_base: bool):
        with open(conf_file, "r") as f:
            self.conf_dict = yaml.safe_load(f)
        self.with_base = with_base

    def get_kin(self) -> KinematicModel:
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()
        kin = KinematicModel(urdf_str)
        return kin

    @property
    @abstractmethod
    def robot_model(self) -> RobotModel:
        ...

    @property
    def urdf_path(self) -> Path:
        return Path(self.conf_dict["urdf_path"]).expanduser()

    @property
    @abstractmethod
    def control_joint_names(self) -> List[str]:
        ...

    @abstractmethod
    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        pass

    @abstractmethod
    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_sphere_specs(self) -> List[SphereAttachentSpec]:
        # the below reads the all the sphere specs from the yaml file
        # but if you want to use the sphere specs for the specific links
        # you can override this method
        d = self.conf_dict["collision_spheres"]
        sphere_specs = []
        for link_name, vals in d.items():
            ignore_collision = vals["ignore_collision"]
            spheres_d = vals["spheres"]
            for spec in spheres_d:
                vals = np.array(spec)
                center, r = vals[:3], vals[3]
                sphere_specs.append(SphereAttachentSpec(link_name, center, r, ignore_collision))
        return sphere_specs

    def create_collision_const(self, self_collision: bool = True) -> SphereCollisionCst:
        sphere_specs = self.get_sphere_specs()
        if self_collision:
            if "self_collision_pairs" not in self.conf_dict:
                raise ValueError("self_collision_pairs is not defined in the yaml file")
            self_collision_pairs = self.conf_dict["self_collision_pairs"]
            sdfs = [sksdf_to_cppsdf(sk.sdf) for sk in self.self_body_collision_primitives()]
        else:
            self_collision_pairs = []
            sdfs = []
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()
        kin = KinematicModel(urdf_str)
        cst = SphereCollisionCst(
            kin, self.control_joint_names, self.with_base, sphere_specs, self_collision_pairs, sdfs
        )
        return cst

    def create_pose_const(self, link_names: List[str], link_poses: List[np.ndarray]) -> LinkPoseCst:
        # read urdf file to str
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()
        kin = KinematicModel(urdf_str)
        return LinkPoseCst(kin, self.control_joint_names, self.with_base, link_names, link_poses)


class FetchSpec(RobotSpec):
    def __init__(self, with_base: bool = False):
        # set with_base = True only in testing
        p = Path(__file__).parent / "conf" / "fetch.yaml"
        super().__init__(p, with_base)

    @property
    def robot_model(self) -> RobotModel:
        return load_urdf_model_using_cache(self.urdf_path)

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        base = Cylinder(0.29, 0.32, face_colors=[255, 255, 255, 200], with_sdf=True)
        base.translate([0.005, 0.0, 0.2])
        torso = Box([0.16, 0.16, 1.0], face_colors=[255, 255, 255, 200], with_sdf=True)
        torso.translate([-0.12, 0.0, 0.5])

        neck_lower = Box([0.1, 0.18, 0.08], face_colors=[255, 255, 255, 200], with_sdf=True)
        neck_lower.translate([0.0, 0.0, 0.97])
        neck_upper = Box([0.05, 0.17, 0.15], face_colors=[255, 255, 255, 200], with_sdf=True)
        neck_upper.translate([-0.035, 0.0, 0.92])

        torso_left = Cylinder(0.1, 1.5, face_colors=[255, 255, 255, 200], with_sdf=True)
        torso_left.translate([-0.143, 0.09, 0.75])
        torso_right = Cylinder(0.1, 1.5, face_colors=[255, 255, 255, 200], with_sdf=True)
        torso_right.translate([-0.143, -0.09, 0.75])

        head = Cylinder(0.235, 0.12, face_colors=[255, 255, 255, 200], with_sdf=True)
        head.translate([0.0, 0.0, 1.04])
        self_body_obstacles = [base, torso, torso_left, torso_right]
        return self_body_obstacles

    def create_gripper_pose_const(self, link_pose: np.ndarray) -> LinkPoseCst:
        return self.create_pose_const(["wrist_roll_link"], [link_pose])

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        # it takes time to parse the urdf file so we do it here...
        min_angles = np.array(
            [0.0, -1.6056, -1.221, -np.pi * 2, -2.251, -np.pi * 2, -2.16, -np.pi * 2]
        )
        max_angles = np.array(
            [0.38615, 1.6056, 1.518, np.pi * 2, 2.251, np.pi * 2, 2.16, np.pi * 2]
        )
        return min_angles, max_angles


class JaxonSpec(RobotSpec):
    def __init__(self):
        p = Path(__file__).parent / "conf" / "jaxon.yaml"
        super().__init__(p)

    def get_kin(self):
        kin = super().get_kin()
        matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])
        rpy = np.flip(rpy_angle(matrix)[0])
        kin.add_new_link("rarm_end_coords", "LARM_LINK7", np.array([0, 0, -0.220]), rpy)
        kin.add_new_link("larm_end_coords", "LARM_LINK7", np.array([0, 0, -0.220]), rpy)
        kin.add_new_link("rleg_end_coords", "RLEG_LINK5", np.array([0, 0, -0.1]), np.zeros(3))
        kin.add_new_link("lleg_end_coords", "LLEG_LINK5", np.array([0, 0, -0.1]), np.zeros(3))
        return kin

    @property
    def robot_model(self) -> RobotModel:
        matrix = rotation_matrix(np.pi * 0.5, [0, 0, 1.0])
        model = load_urdf_model_using_cache(self.urdf_path)

        model.rarm_end_coords = CascadedCoords(self.RARM_LINK7, name="rarm_end_coords")
        model.rarm_end_coords.translate([0, 0, -0.220])
        model.rarm_end_coords.rotate_with_matrix(matrix, wrt="local")

        model.rarm_tip_coords = CascadedCoords(self.RARM_LINK7, name="rarm_end_coords")
        model.rarm_tip_coords.translate([0, 0, -0.3])
        model.rarm_tip_coords.rotate_with_matrix(matrix, wrt="local")

        model.larm_end_coords = CascadedCoords(self.LARM_LINK7, name="larm_end_coords")
        model.larm_end_coords.translate([0, 0, -0.220])
        model.larm_end_coords.rotate_with_matrix(matrix, wrt="local")

        model.rleg_end_coords = CascadedCoords(self.RLEG_LINK5, name="rleg_end_coords")
        model.rleg_end_coords.translate([0, 0, -0.1])

        model.lleg_end_coords = CascadedCoords(self.LLEG_LINK5, name="lleg_end_coords")
        model.lleg_end_coords.translate([0, 0, -0.1])
        return model

    @property
    def control_joint_names(self) -> List[str]:
        return self.conf_dict["control_joint_names"]

    def get_sphere_specs(self) -> List[SphereAttachentSpec]:
        # because legs are on the ground, we don't need to consider the spheres on the legs
        specs = super().get_sphere_specs()
        filtered = []
        for spec in specs:
            if spec.parent_link_name in ("RLEG_LINK5", "LLEG_LINK5"):
                continue
            filtered.append(spec)
        return filtered

    def self_body_collision_primitives(self) -> Sequence[Union[Box, Sphere, Cylinder]]:
        return []

    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
