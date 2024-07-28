from _fused import FusedSpheresCollisionChecker, fuck, Pose, BoxSDF
import tinyfk
import numpy as np

pose = Pose(np.ones(3), np.eye(3))
sdf = BoxSDF([1, 1, 1], pose)

urdf_model_path = tinyfk.pr2_urdfpath()
joint_names = [
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_forearm_roll_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
]
kin = FusedSpheresCollisionChecker(urdf_model_path, joint_names, [], [], [], sdf)
ret = kin.is_valid(np.random.randn(7))
print(ret)


