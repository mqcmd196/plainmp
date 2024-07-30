#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>
#include <tinyfk.hpp>
#include <utility>
#include "primitive_sdf.hpp"

namespace cst {

namespace py = pybind11;
using namespace primitive_sdf;

struct SphereAttachentSpec {
  std::string parent_link_name;
  Eigen::Vector3d relative_position;
  double radius;
  bool ignore_collision;
};

class FusedSpheresCollisionChecker {
 public:
  FusedSpheresCollisionChecker(
      const std::string& urdf_string,
      const std::vector<std::string>& control_joint_names,
      const std::vector<SphereAttachentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      const std::vector<PrimitiveSDFBase::Ptr>& fixed_sdfs);

  bool is_valid(const std::vector<double>& q);
  double evaluate(const std::vector<double>& q) const;

 private:
  std::vector<size_t> sphere_ids_;
  std::shared_ptr<tinyfk::KinematicModel> kin_;
  std::vector<size_t> control_joint_ids_;
  std::vector<SphereAttachentSpec> sphere_specs_;
  std::vector<std::pair<size_t, size_t>> selcol_pairs_ids_;
  std::vector<PrimitiveSDFBase::Ptr> fixed_sdfs_;
};

void bind_collision_constraints(py::module& m);

};  // namespace cst
