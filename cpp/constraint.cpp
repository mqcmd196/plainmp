#include "constraint.hpp"
#include <pybind11/stl.h>
#include <optional>
#include "primitive_sdf.hpp"

namespace cst {

namespace py = pybind11;

FusedSpheresCollisionChecker::FusedSpheresCollisionChecker(
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

bool FusedSpheresCollisionChecker::is_valid(const std::vector<double>& q) {
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

/*
double FusedSpheresCollisionChecker::evaluate(
    const std::vector<double>& q) const {
  kin_->set_joint_angles(control_joint_ids_, q);
  tinyfk::Transform pose;

  double min_val = std::numeric_limits<double>::max();
  std::optional<size_t> min_link_id = std::nullopt;
  std::optional<size_t> min_sdf_id = std::nullopt;  // nullopt for fixed sdf
  std::optional<bool> min_is_attached = std::nullopt;
  std::shared_ptr<primitive_sdf::PrimitiveSDFBase::Ptr> min_sdf = nullptr;

  for (size_t i = 0; i < sphere_ids_.size(); i++) {
    // vs fixed sdf
    kin_->get_link_pose(sphere_ids_[i], pose);
    Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
    for(auto& sdf : fixed_sdfs_) {
      double val = sdf->evaluate(center) - sphere_specs_[i].radius;
      if(val < min_val) {
        min_val = val;
        min_link_id = sphere_ids_[i];
        min_sdf_id = std::nullopt;
        min_is_attached = false;
        min_sdf = sdf;
      }
    }
    // vs attached sdf
    for (size_t j = 0; j < sdf_ids_.size(); j++) {
      kin_->get_link_pose(sdf_ids_[j], pose);
      Eigen::Vector3d center_sdf(pose.position.x, pose.position.y,
                                 pose.position.z);
      sdf_specs_[j].sdf->tf_.set_position(center_sdf);  // TODO orientation
      double val = sdf_specs_[j].sdf->evaluate(center, sphere_specs_[i].radius);
      if(val < min_val) {
        min_val = val;
        min_link_id = sphere_ids_[i];
        min_sdf_id = sdf_ids_[j];
        min_is_attached = true;
        min_sdf = sdf_specs_[j].sdf;
      }
    }
  }
  // compute jacobian
  kin_->get_link_pose(min_link_id, pose);
  Eigen::Vector3d sphere_pos;
  sphere_pos << pose.position.x, pose.position.y, pose.position.z;

  // compute gradient
  if(min_is_attached.value()) {
    // kin_->get_link_pose(min_link_id, pose);
    // sphere_pos << pose.position.x, pose.position.y, pose.position.z;

    // kin_->get_link_pose(min_sdf_id, pose);
    // sdf_pos << pose.position.x, pose.position.y, pose.position.z;
    // min_sdf->tf_.set_position(sdf_pos);
    // auto jac_sdf = kin_->get_jacobian(min_sdf_id, control_joint_ids_);

    // grad = (jac_sphere - jac_sdf) * (sphere_pos - sdf_pos)
  } else {

  }
}
*/

void bind_collision_constraints(py::module& m) {
  py::class_<SphereAttachentSpec>(m, "SphereAttachentSpec")
      .def(
          py::init<const std::string&, const Eigen::Vector3d&, double, bool>());

  py::class_<FusedSpheresCollisionChecker>(m, "FusedSpheresCollisionChecker")
      .def(py::init<const std::string&, const std::vector<std::string>&,
                    const std::vector<SphereAttachentSpec>&,
                    const std::vector<std::pair<std::string, std::string>>&,
                    const std::vector<primitive_sdf::PrimitiveSDFBase::Ptr>&>())
      .def("is_valid", &FusedSpheresCollisionChecker::is_valid);
}

}  // namespace cst
