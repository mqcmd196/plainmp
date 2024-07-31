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

class ConstraintBase {
 public:
  virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(
      const std::vector<double>& q) const = 0;
  virtual size_t cst_dim() const = 0;
  virtual bool is_equality() const = 0;
  virtual ~ConstraintBase() = default;
};

class EqConstraintBase : public ConstraintBase {
 public:
  bool is_equality() const override { return true; }
};

class IneqConstraintBase : public ConstraintBase {
 public:
  virtual bool is_valid(const std::vector<double>& q) = 0;
  bool is_equality() const override { return false; }
};

class LinkPoseCst : EqConstraintBase {
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
    return dim;
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

class SphereCollisionCst : IneqConstraintBase {
 public:
  SphereCollisionCst(
      std::shared_ptr<tinyfk::KinematicModel> kin,
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

  size_t cst_dim() const {
    if (selcol_pairs_ids_.size() == 0) {
      return 1;
    } else {
      return 2;
    }
  }

 private:
  std::vector<PrimitiveSDFBase::Ptr> get_all_sdfs() const {
    // TODO: Consider using std::views::concat (but it's C++20)
    std::vector<PrimitiveSDFBase::Ptr> all_sdfs = fixed_sdfs_;
    all_sdfs.insert(all_sdfs.end(), sdfs_.begin(), sdfs_.end());
    if (all_sdfs.size() == 0) {
      throw std::runtime_error("(cpp) No SDFs are set");
    }
    return all_sdfs;
  }

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
