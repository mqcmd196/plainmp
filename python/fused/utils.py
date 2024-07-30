import fused.psdf as psdf
from skrobot.sdf import BoxSDF, CylinderSDF, UnionSDF


def sksdf_to_cppsdf(sksdf) -> psdf.SDFBase:
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
        cpp_sdf_list = [sksdf_to_cppsdf(s) for s in sksdf.sdf_list]
        sdf = psdf.UnionSDF(cpp_sdf_list)
    else:
        raise ValueError(f"Unsupported SDF type {type(sksdf)}")
    return sdf
