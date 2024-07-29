#include "primitive_sdf.hpp"
#include <tinyfk.hpp>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
        ) 
      : 
        kin_(std::make_shared<tinyfk::KinematicModel>(tinyfk::load_urdf(urdf_string))),
        control_joint_ids_(kin_->get_joint_ids(control_joint_names)),
        sphere_specs_(sphere_specs),
        sdf_specs_(sdf_specs),
        sdf_fixed_(sdf_fixed)
  {
    for(const auto& spec : sphere_specs) {
      auto parent_id = kin_->get_link_ids({spec.parent_link_name})[0];
      auto name = "sphere" + std::to_string(sphere_ids_.size()); 
      kin_->add_new_link(name, parent_id, 
          {spec.relative_position.x(), spec.relative_position.y(), spec.relative_position.z()},
          {0.0, 0.0, 0.0});
      sphere_ids_.push_back(kin_->get_link_ids({name})[0]);
    }

    for(const auto& spec : sdf_specs) {
      auto parent_id = kin_->get_link_ids({spec.parent_link_name})[0];
      auto name = "sdf" + std::to_string(sdf_ids_.size()); 
      kin_->add_new_link(name, parent_id, 
          {spec.relative_position.x(), spec.relative_position.y(), spec.relative_position.z()},
          {0.0, 0.0, 0.0});
      sdf_ids_.push_back(kin_->get_link_ids({name})[0]);
    }
  }

  bool is_valid(const std::vector<double>& q) {
    kin_->set_joint_angles(control_joint_ids_, q);

    tinyfk::Transform pose;
    for(size_t i = 0; i < sphere_ids_.size(); i++) {
      // vs fixed sdf
      kin_->get_link_pose(sphere_ids_[i], pose);
      Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
      if(!sdf_fixed_->is_outside(center, sphere_specs_[i].radius)) {
        return false;
      }
      // vs attached sdf
      for(size_t j = 0; j < sdf_ids_.size(); j++) {
        kin_->get_link_pose(sdf_ids_[j], pose);
        Eigen::Vector3d center_sdf(pose.position.x, pose.position.y, pose.position.z);
        sdf_specs_[j].sdf->tf_.set_position(center_sdf); // TODO orientation
        if(!sdf_specs_[j].sdf->is_outside(center, sphere_specs_[i].radius)) {
          return false;
        }
      }
    }
    return true;
  }

  private:
    std::vector<size_t> sphere_ids_;
    std::vector<size_t> sdf_ids_;
    std::shared_ptr<tinyfk::KinematicModel> kin_;
    std::vector<size_t> control_joint_ids_;
    std::vector<SphereAttachentSpec> sphere_specs_;
    std::vector<SDFAttachmentSpec> sdf_specs_;
    primitive_sdf::SDFBase::Ptr sdf_fixed_;
};

void bind_collision_constraints(py::module &m) {
  py::class_<SphereAttachentSpec >(m, "SphereAttachentSpec")
      .def(py::init<const std::string&, const Eigen::Vector3d&, double>());

  py::class_<SDFAttachmentSpec >(m, "SDFAttachmentSpec")
    .def(py::init<const std::string&, const Eigen::Vector3d&, PrimitiveSDFBase::Ptr>());

  py::class_<FusedSpheresCollisionChecker>(m, "FusedSpheresCollisionChecker")
      .def(py::init<const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<SphereAttachentSpec>&,
                    const std::vector<SDFAttachmentSpec>&,
                    primitive_sdf::SDFBase::Ptr>())
      .def("is_valid", &FusedSpheresCollisionChecker::is_valid);
}
