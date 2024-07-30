#include "constraint.hpp"
#include <pybind11/stl.h>
#include <optional>
#include "primitive_sdf.hpp"
#include "tinyfk.hpp"

namespace cst {

namespace py = pybind11;

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LinkPoseCst::evaluate(
    const std::vector<double>& q) const {
  kin_->set_joint_angles(control_joint_ids_, q);
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), control_joint_ids_.size());
  tinyfk::Transform pose;
  size_t head = 0;
  for (size_t i = 0; i < link_ids_.size(); i++) {
    kin_->get_link_pose(link_ids_[i], pose);
    if (poses_[i].size() == 3) {
      vals.segment(head, 3) =
          Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z) -
          poses_[i];
      jac.block(head, 0, 3, control_joint_ids_.size()) = kin_->get_jacobian(
          link_ids_[i], control_joint_ids_, tinyfk::RotationType::IGNORE);
      head += 3;
    } else {
      vals.segment(head, 3) =
          Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z) -
          poses_[i];
      auto rpy = pose.rotation.getRPY();
      vals.segment(head + 3, 3) =
          Eigen::Vector3d(rpy.x, rpy.y, rpy.z) - poses_[i].segment(3, 3);
      jac.block(head, 0, 3, control_joint_ids_.size()) = kin_->get_jacobian(
          link_ids_[i], control_joint_ids_, tinyfk::RotationType::RPY);
      head += 6;
    }
  }
  return {vals, jac};
}

SphereCollisionCst::SphereCollisionCst(
    const std::string& urdf_string,
    const std::vector<std::string>& control_joint_names,
    const std::vector<SphereAttachentSpec>& sphere_specs,
    const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
    const std::vector<primitive_sdf::PrimitiveSDFBase::Ptr>& fixed_sdfs)
    : kin_(std::make_shared<tinyfk::KinematicModel>(
          tinyfk::load_urdf(urdf_string))),
      control_joint_ids_(kin_->get_joint_ids(control_joint_names)),
      sphere_specs_(sphere_specs),
      fixed_sdfs_(fixed_sdfs) {
  std::vector<std::string> parent_link_names;
  for (const auto& spec : sphere_specs) {
    auto parent_id = kin_->get_link_ids({spec.parent_link_name})[0];
    auto name = "sphere" + std::to_string(sphere_ids_.size());
    kin_->add_new_link(name, parent_id,
                       {spec.relative_position.x(), spec.relative_position.y(),
                        spec.relative_position.z()},
                       {0.0, 0.0, 0.0});
    sphere_ids_.push_back(kin_->get_link_ids({name})[0]);
    parent_link_names.push_back(spec.parent_link_name);
  }
  std::vector<std::pair<size_t, size_t>> selcol_pairs_ids;
  for (const auto& pair : selcol_pairs) {
    std::vector<size_t> first_ids;
    std::vector<size_t> second_ids;
    for (size_t i = 0; i < parent_link_names.size(); i++) {
      if (parent_link_names[i] == pair.first) {
        first_ids.push_back(i);
      }
      if (parent_link_names[i] == pair.second) {
        second_ids.push_back(i);
      }
    }
    for (auto& first_id : first_ids) {
      for (auto& second_id : second_ids) {
        selcol_pairs_ids.push_back({first_id, second_id});
      }
    }
  }
  selcol_pairs_ids_ = selcol_pairs_ids;
}

bool SphereCollisionCst::is_valid(const std::vector<double>& q) {
  kin_->set_joint_angles(control_joint_ids_, q);
  tinyfk::Transform pose;
  for (size_t i = 0; i < sphere_ids_.size(); i++) {
    if (sphere_specs_[i].ignore_collision) {
      continue;
    }
    kin_->get_link_pose(sphere_ids_[i], pose);
    Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
    for (auto& sdf : fixed_sdfs_) {
      if (!sdf->is_outside(center, sphere_specs_[i].radius)) {
        return false;
      }
    }
    for (auto& sdf : sdfs_) {
      if (!sdf->is_outside(center, sphere_specs_[i].radius)) {
        return false;
      }
    }
  }
  tinyfk::Transform pose1, pose2;
  for (const auto& pair : selcol_pairs_ids_) {
    kin_->get_link_pose(sphere_ids_[pair.first], pose1);
    kin_->get_link_pose(sphere_ids_[pair.second], pose2);
    Eigen::Vector3d center1(pose1.position.x, pose1.position.y,
                            pose1.position.z);
    Eigen::Vector3d center2(pose2.position.x, pose2.position.y,
                            pose2.position.z);
    if ((center1 - center2).norm() <
        sphere_specs_[pair.first].radius + sphere_specs_[pair.second].radius) {
      return false;
    }
  }
  return true;
}
std::pair<Eigen::VectorXd, Eigen::MatrixXd> SphereCollisionCst::evaluate(
    const std::vector<double>& q) const {
  kin_->set_joint_angles(control_joint_ids_, q);

  // collision vs outers
  tinyfk::Transform pose;
  Eigen::VectorXd grad_in_cspace_other(control_joint_ids_.size());
  double min_val_other = std::numeric_limits<double>::max();
  std::optional<size_t> min_sphere_idx = std::nullopt;
  std::optional<size_t> min_sdf_idx = std::nullopt;
  {
    for (size_t i = 0; i < sphere_ids_.size(); i++) {
      if (sphere_specs_[i].ignore_collision) {
        continue;
      }
      kin_->get_link_pose(sphere_ids_[i], pose);
      Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
      for (size_t j = 0; j < fixed_sdfs_.size(); j++) {
        double val = fixed_sdfs_[j]->evaluate(center) - sphere_specs_[i].radius;
        if (val < min_val_other) {
          min_val_other = val;
          min_sphere_idx = i;
          min_sdf_idx = j;
        }
      }
      for (size_t j = 0; j < sdfs_.size(); j++) {
        double val = sdfs_[j]->evaluate(center) - sphere_specs_[i].radius;
        if (val < min_val_other) {
          min_val_other = val;
          min_sphere_idx = i;
          min_sdf_idx = j;
        }
      }
    }

    Eigen::Vector3d grad;
    kin_->get_link_pose(sphere_ids_[*min_sphere_idx], pose);
    Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
    for (size_t i = 0; i < 3; i++) {
      Eigen::Vector3d perturbed_center = center;
      perturbed_center[i] += 1e-6;
      double val = fixed_sdfs_[*min_sdf_idx]->evaluate(perturbed_center) -
                   sphere_specs_[*min_sphere_idx].radius;
      grad[i] = (val - min_val_other) / 1e-6;
    }
    auto sphere_jac =
        kin_->get_jacobian(sphere_ids_[*min_sphere_idx], control_joint_ids_);
    grad_in_cspace_other = sphere_jac.transpose() * grad;
  }

  // collision vs inners (self collision)
  Eigen::VectorXd grad_in_cspace_self(control_joint_ids_.size());
  double min_val_self = std::numeric_limits<double>::max();
  {
    std::optional<std::pair<size_t, size_t>> min_pair = std::nullopt;
    for (const auto& pair : selcol_pairs_ids_) {
      kin_->get_link_pose(sphere_ids_[pair.first], pose);
      Eigen::Vector3d center1(pose.position.x, pose.position.y,
                              pose.position.z);
      kin_->get_link_pose(sphere_ids_[pair.second], pose);
      Eigen::Vector3d center2(pose.position.x, pose.position.y,
                              pose.position.z);
      double val = (center1 - center2).norm() -
                   sphere_specs_[pair.first].radius -
                   sphere_specs_[pair.second].radius;
      if (val < min_val_self) {
        min_val_self = val;
        min_pair = pair;
      }
    }
    Eigen::Vector3d center1, center2;
    kin_->get_link_pose(sphere_ids_[min_pair->first], pose);
    center1 << pose.position.x, pose.position.y, pose.position.z;
    kin_->get_link_pose(sphere_ids_[min_pair->second], pose);
    center2 << pose.position.x, pose.position.y, pose.position.z;
    Eigen::MatrixXd&& jac1 =
        kin_->get_jacobian(sphere_ids_[min_pair->first], control_joint_ids_);
    Eigen::MatrixXd&& jac2 =
        kin_->get_jacobian(sphere_ids_[min_pair->second], control_joint_ids_);
    double norminv = 1.0 / (center1 - center2).norm();
    grad_in_cspace_self =
        norminv * (center1 - center2).transpose() * (jac1 - jac2);
  }

  Eigen::Vector2d vals(min_val_other, min_val_self);
  Eigen::MatrixXd jac(2, grad_in_cspace_other.size());
  jac.row(0) = grad_in_cspace_other;
  jac.row(1) = grad_in_cspace_self;
  return {vals, jac};
}

void bind_collision_constraints(py::module& m) {
  auto cst_m = m.def_submodule("constraint");
  py::class_<LinkPoseCst>(cst_m, "LinkPoseCst")
      .def(py::init<const std::string&, const std::vector<std::string>&,
                    const std::vector<Eigen::VectorXd>&>())
      .def("evaluate", &LinkPoseCst::evaluate)
      .def("cst_dim", &LinkPoseCst::cst_dim);
  py::class_<SphereAttachentSpec>(cst_m, "SphereAttachentSpec")
      .def(
          py::init<const std::string&, const Eigen::Vector3d&, double, bool>());

  py::class_<SphereCollisionCst>(cst_m, "SphereCollisionCst")
      .def(py::init<const std::string&, const std::vector<std::string>&,
                    const std::vector<SphereAttachentSpec>&,
                    const std::vector<std::pair<std::string, std::string>>&,
                    const std::vector<primitive_sdf::PrimitiveSDFBase::Ptr>&>())
      .def("is_valid", &SphereCollisionCst::is_valid)
      .def("evaluate", &SphereCollisionCst::evaluate);
}

}  // namespace cst
