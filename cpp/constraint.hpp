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

class LinkPoseCst {
 public:
  LinkPoseCst(std::shared_ptr<tinyfk::KinematicModel> kin,
              const std::vector<std::string>& control_joint_names,
              const std::vector<std::string>& link_names,
              const std::vector<Eigen::VectorXd>& poses)
      : kin_(kin),
        control_joint_ids_(kin_->get_joint_ids(control_joint_names)),
        link_ids_(kin_->get_link_ids(link_names)),
        poses_(poses) {
    for (auto& pose : poses_) {
      if (pose.size() != 3 && pose.size() != 6) {
        throw std::runtime_error("All poses must be 3 or 6 dimensional");
      }
    }
  }
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(
      const std::vector<double>& q) const;
  size_t cst_dim() const {
    size_t dim = 0;
    for (auto& pose : poses_) {
      dim += pose.size();
    }
  }

 private:
  std::shared_ptr<tinyfk::KinematicModel> kin_;
  std::vector<size_t> control_joint_ids_;
  std::vector<size_t> link_ids_;
  std::vector<Eigen::VectorXd> poses_;
};

struct SphereAttachentSpec {
  std::string parent_link_name;
  Eigen::Vector3d relative_position;
  double radius;
  bool ignore_collision;
};

class SphereCollisionCst {
 public:
  SphereCollisionCst(
      const std::string& urdf_string,
      const std::vector<std::string>& control_joint_names,
      const std::vector<SphereAttachentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      const std::vector<PrimitiveSDFBase::Ptr>& fixed_sdfs);

  void set_sdfs(const std::vector<PrimitiveSDFBase::Ptr>& sdfs) {
    sdfs_ = sdfs;
  }

  bool is_valid(const std::vector<double>& q);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(
      const std::vector<double>& q) const;

 private:
  std::vector<size_t> sphere_ids_;
  std::shared_ptr<tinyfk::KinematicModel> kin_;
  std::vector<size_t> control_joint_ids_;
  std::vector<SphereAttachentSpec> sphere_specs_;
  std::vector<std::pair<size_t, size_t>> selcol_pairs_ids_;
  std::vector<PrimitiveSDFBase::Ptr> fixed_sdfs_;  // fixed after construction
  std::vector<PrimitiveSDFBase::Ptr> sdfs_;        // set later by user
};

void bind_collision_constraints(py::module& m);

};  // namespace cst
