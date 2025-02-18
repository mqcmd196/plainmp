#include "constraint.hpp"
#include <pybind11/stl.h>
#include <optional>
#include <stdexcept>
#include "primitive_sdf.hpp"
#include "tinyfk.hpp"

namespace cst {

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LinkPoseCst::evaluate_dirty()
    const {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());
  tinyfk::Transform pose;
  size_t head = 0;
  for (size_t i = 0; i < link_ids_.size(); i++) {
    kin_->get_link_pose(link_ids_[i], pose);
    if (poses_[i].size() == 3) {
      vals.segment(head, 3) =
          Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z) -
          poses_[i];
      jac.block(head, 0, 3, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
      head += 3;
    } else if (poses_[i].size() == 6) {
      vals.segment(head, 3) =
          Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z) -
          poses_[i];
      auto rpy = pose.rotation.getRPY();
      vals.segment(head + 3, 3) =
          Eigen::Vector3d(rpy.x, rpy.y, rpy.z) - poses_[i].segment(3, 3);
      jac.block(head, 0, 6, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             tinyfk::RotationType::RPY, with_base_);
      head += 6;
    } else {
      vals.segment(head, 3) =
          Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z) -
          poses_[i];
      vals[head + 3] = pose.rotation.x - poses_[i][3];
      vals[head + 4] = pose.rotation.y - poses_[i][4];
      vals[head + 5] = pose.rotation.z - poses_[i][5];
      vals[head + 6] = pose.rotation.w - poses_[i][6];
      jac.block(head, 0, 7, q_dim()) =
          kin_->get_jacobian(link_ids_[i], control_joint_ids_,
                             tinyfk::RotationType::XYZW, with_base_);
      head += 7;
    }
  }
  return {vals, jac};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> RelativePoseCst::evaluate_dirty()
    const {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());
  tinyfk::Transform pose_dummy, pose2;
  kin_->get_link_pose(dummy_link_id_, pose_dummy);
  kin_->get_link_pose(link_id2_, pose2);
  vals[0] = pose_dummy.position.x - pose2.position.x;
  vals[1] = pose_dummy.position.y - pose2.position.y;
  vals[2] = pose_dummy.position.z - pose2.position.z;
  vals[3] = pose_dummy.rotation.x - pose2.rotation.x;
  vals[4] = pose_dummy.rotation.y - pose2.rotation.y;
  vals[5] = pose_dummy.rotation.z - pose2.rotation.z;
  vals[6] = pose_dummy.rotation.w - pose2.rotation.w;

  jac = kin_->get_jacobian(dummy_link_id_, control_joint_ids_,
                           tinyfk::RotationType::XYZW, with_base_) -
        kin_->get_jacobian(link_id2_, control_joint_ids_,
                           tinyfk::RotationType::XYZW, with_base_);
  return {vals, jac};
}

SphereCollisionCst::SphereCollisionCst(
    std::shared_ptr<tinyfk::KinematicModel> kin,
    const std::vector<std::string>& control_joint_names,
    bool with_base,
    const std::vector<SphereAttachmentSpec>& sphere_specs,
    const std::vector<std::pair<std::string, std::string>>& selcol_pairs,
    std::optional<SDFBase::Ptr> fixed_sdf)
    : IneqConstraintBase(kin, control_joint_names, with_base),
      sphere_specs_(sphere_specs),
      fixed_sdf_(fixed_sdf == std::nullopt ? nullptr : *fixed_sdf) {
  std::vector<std::string> parent_link_names;
  for (const auto& spec : sphere_specs) {
    auto parent_id = kin_->get_link_ids({spec.parent_link_name})[0];
    kin_->add_new_link(spec.name, parent_id,
                       {spec.relative_position.x(), spec.relative_position.y(),
                        spec.relative_position.z()},
                       {0.0, 0.0, 0.0});
    sphere_ids_.push_back(kin_->get_link_ids({spec.name})[0]);
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

bool SphereCollisionCst::is_valid_dirty() const {
  tinyfk::Transform pose;
  for (size_t i = 0; i < sphere_ids_.size(); i++) {
    if (sphere_specs_[i].ignore_collision) {
      continue;
    }
    kin_->get_link_pose(sphere_ids_[i], pose);
    Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
    for (auto& sdf : get_all_sdfs()) {
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

std::pair<Eigen::VectorXd, Eigen::MatrixXd> SphereCollisionCst::evaluate_dirty()
    const {
  auto all_sdfs = get_all_sdfs();

  // collision vs outers
  tinyfk::Transform pose;
  Eigen::VectorXd grad_in_cspace_other(q_dim());
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
      for (size_t j = 0; j < all_sdfs.size(); j++) {
        double val = all_sdfs[j]->evaluate(center) - sphere_specs_[i].radius;
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
      double val = all_sdfs[*min_sdf_idx]->evaluate(perturbed_center) -
                   sphere_specs_[*min_sphere_idx].radius;
      grad[i] = (val - min_val_other) / 1e-6;
    }
    auto sphere_jac =
        kin_->get_jacobian(sphere_ids_[*min_sphere_idx], control_joint_ids_,
                           tinyfk::RotationType::IGNORE, with_base_);
    grad_in_cspace_other = sphere_jac.transpose() * grad;
  }

  if (selcol_pairs_ids_.size() == 0) {
    Eigen::MatrixXd jac(1, grad_in_cspace_other.size());
    jac.row(0) = grad_in_cspace_other;
    return {Eigen::VectorXd::Constant(1, min_val_other), jac};
  } else {
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
          kin_->get_jacobian(sphere_ids_[min_pair->first], control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
      Eigen::MatrixXd&& jac2 =
          kin_->get_jacobian(sphere_ids_[min_pair->second], control_joint_ids_,
                             tinyfk::RotationType::IGNORE, with_base_);
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
}

bool ComInPolytopeCst::is_valid_dirty() const {
  // COPIED from evaluate() >> START
  auto com_tmp = kin_->get_com();
  Eigen::Vector3d com(com_tmp.x, com_tmp.y, com_tmp.z);
  if (force_link_ids_.size() > 0) {
    double vertical_force_sum = 1.0;  // 1.0 for normalized self
    tinyfk::Transform pose;
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      double force = applied_force_values_[j] / kin_->total_mass_;
      vertical_force_sum += force;
      kin_->get_link_pose(force_link_ids_[j], pose);
      Eigen::Vector3d force_pos(pose.position.x, pose.position.y,
                                pose.position.z);
      com += force * force_pos;
    }
    com /= vertical_force_sum;
  }
  // COPIED from evaluate() >> END
  return polytope_sdf_->evaluate(com) < 0;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ComInPolytopeCst::evaluate_dirty()
    const {
  Eigen::VectorXd vals(cst_dim());
  Eigen::MatrixXd jac(cst_dim(), q_dim());

  auto com_tmp = kin_->get_com();
  Eigen::Vector3d com(com_tmp.x, com_tmp.y, com_tmp.z);

  auto com_jaco = kin_->get_com_jacobian(control_joint_ids_, q_dim());
  if (force_link_ids_.size() > 0) {
    double vertical_force_sum = 1.0;  // 1.0 for normalized self
    tinyfk::Transform pose;
    for (size_t j = 0; j < force_link_ids_.size(); ++j) {
      double force = applied_force_values_[j] / kin_->total_mass_;
      vertical_force_sum += force;
      kin_->get_link_pose(force_link_ids_[j], pose);
      Eigen::Vector3d force_pos(pose.position.x, pose.position.y,
                                pose.position.z);
      com += force * force_pos;

      com_jaco += kin_->get_jacobian(force_link_ids_[j], control_joint_ids_,
                                     tinyfk::RotationType::IGNORE, with_base_) *
                  force;
    }
    double inv = 1.0 / vertical_force_sum;
    com *= inv;
    com_jaco *= inv;
  }
  double val = -polytope_sdf_->evaluate(com);
  vals[0] = val;

  Eigen::Vector3d grad;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d perturbed_com = com;
    perturbed_com[i] += 1e-6;
    double val_perturbed = -polytope_sdf_->evaluate(perturbed_com);
    grad[i] = (val_perturbed - val) / 1e-6;
  }
  jac.row(0) = com_jaco.transpose() * grad;

  return {vals, jac};
};

}  // namespace cst
