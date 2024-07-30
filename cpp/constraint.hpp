#include "primitive_sdf.hpp"
#include <tinyfk.hpp>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace cst {

namespace py = pybind11;
using namespace primitive_sdf;

struct SphereAttachentSpec {
  std::string parent_link_name;
  Eigen::Vector3d relative_position;
  double radius;
};

struct SDFAttachmentSpec {
  std::string parent_link_name;
  Eigen::Vector3d relative_position;
  PrimitiveSDFBase::Ptr sdf;
};

class FusedSpheresCollisionChecker {
  public:
    FusedSpheresCollisionChecker(
        const std::string& urdf_string,
        const std::vector<std::string>& control_joint_names,
        const std::vector<SphereAttachentSpec>& sphere_specs,
        const std::vector<SDFAttachmentSpec>& sdf_specs,
        primitive_sdf::SDFBase::Ptr sdf_fixed
        ); 
  bool is_valid(const std::vector<double>& q);

  private:
    std::vector<size_t> sphere_ids_;
    std::vector<size_t> sdf_ids_;
    std::shared_ptr<tinyfk::KinematicModel> kin_;
    std::vector<size_t> control_joint_ids_;
    std::vector<SphereAttachentSpec> sphere_specs_;
    std::vector<SDFAttachmentSpec> sdf_specs_;
    primitive_sdf::SDFBase::Ptr sdf_fixed_;
};

void bind_collision_constraints(py::module &m);

};
