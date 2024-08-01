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
  using Ptr = std::shared_ptr<ConstraintBase>;
  ConstraintBase(std::shared_ptr<tinyfk::KinematicModel> kin,
                 const std::vector<std::string>& control_joint_names)
      : kin_(kin),
        control_joint_ids_(kin->get_joint_ids(control_joint_names)) {}

  void update_kintree(const std::vector<double>& q) {
    kin_->set_joint_angles(control_joint_ids_, q);
  }

  virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate() const = 0;
  virtual size_t cst_dim() const = 0;
  virtual bool is_equality() const = 0;
  virtual ~ConstraintBase() = default;

 protected:
  std::shared_ptr<tinyfk::KinematicModel> kin_;
  std::vector<size_t> control_joint_ids_;
};

class EqConstraintBase : public ConstraintBase {
 public:
  using Ptr = std::shared_ptr<EqConstraintBase>;
  using ConstraintBase::ConstraintBase;
  bool is_equality() const override { return true; }
};

class IneqConstraintBase : public ConstraintBase {
 public:
  using Ptr = std::shared_ptr<IneqConstraintBase>;
  using ConstraintBase::ConstraintBase;
  virtual bool is_valid() const = 0;
  bool is_equality() const override { return false; }
};

class LinkPoseCst : public EqConstraintBase {
 public:
  using Ptr = std::shared_ptr<LinkPoseCst>;
  LinkPoseCst(std::shared_ptr<tinyfk::KinematicModel> kin,
              const std::vector<std::string>& control_joint_names,
              const std::vector<std::string>& link_names,
              const std::vector<Eigen::VectorXd>& poses)
      : EqConstraintBase(kin, control_joint_names),
        link_ids_(kin_->get_link_ids(link_names)),
        poses_(poses) {
    for (auto& pose : poses_) {
      if (pose.size() != 3 && pose.size() != 6) {
        throw std::runtime_error("All poses must be 3 or 6 dimensional");
      }
    }
  }
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate() const override;
  size_t cst_dim() const {
    size_t dim = 0;
    for (auto& pose : poses_) {
      dim += pose.size();
    }
    return dim;
  }

 private:
  std::vector<size_t> link_ids_;
  std::vector<Eigen::VectorXd> poses_;
};

struct SphereAttachentSpec {
  std::string parent_link_name;
  Eigen::Vector3d relative_position;
  double radius;
  bool ignore_collision;
};

class SphereCollisionCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<SphereCollisionCst>;
  SphereCollisionCst(
      std::shared_ptr<tinyfk::KinematicModel> kin,
      const std::vector<std::string>& control_joint_names,
      const std::vector<SphereAttachentSpec>& sphere_specs,
      const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
      const std::vector<PrimitiveSDFBase::Ptr>& fixed_sdfs);

  void set_sdfs(const std::vector<PrimitiveSDFBase::Ptr>& sdfs) {
    sdfs_ = sdfs;
  }

  bool is_valid() const override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate() const override;

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
  std::vector<SphereAttachentSpec> sphere_specs_;
  std::vector<std::pair<size_t, size_t>> selcol_pairs_ids_;
  std::vector<PrimitiveSDFBase::Ptr> fixed_sdfs_;  // fixed after construction
  std::vector<PrimitiveSDFBase::Ptr> sdfs_;        // set later by user
};

class ComInPolytopeCst : public IneqConstraintBase {
 public:
  using Ptr = std::shared_ptr<ComInPolytopeCst>;
  ComInPolytopeCst(std::shared_ptr<tinyfk::KinematicModel> kin,
                   const std::vector<std::string>& control_joint_names,
                   BoxSDF::Ptr polytope_sdf)
      : IneqConstraintBase(kin, control_joint_names),
        polytope_sdf_(polytope_sdf) {
    polytope_sdf_->width_[2] = 1000;  // adhoc to represent infinite height
  }

  bool is_valid() const override;
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate() const override;

  size_t cst_dim() const { return 1; }

 private:
  BoxSDF::Ptr polytope_sdf_;
};

void bind_collision_constraints(py::module& m);

};  // namespace cst
