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

// struct SphereCollection {
// };

class FusedSpheresCollisionChecker {
  public:
    FusedSpheresCollisionChecker(
        //std::shared_ptr<tinyfk::KinematicModel> kin,
        const std::string& urdf_string,
        const std::vector<std::string>& control_joint_names,
        const std::vector<std::string>& parent_link_names,
        const std::vector<Eigen::Vector3d>& sphere_centers,
        const std::vector<double>& sphere_radii,
        primitive_sdf::SDFBase::Ptr sdf
        ) 
      : 
        kin_(std::make_shared<tinyfk::KinematicModel>(tinyfk::load_urdf(urdf_string))),
        control_joint_ids_(kin_->get_joint_ids(control_joint_names)),
        parent_link_ids_(kin_->get_link_ids(parent_link_names)),
        sphere_centers_(sphere_centers),
        sphere_radii_(sphere_radii),
        sdf_(sdf)
  {
    auto parent_ids = kin_->get_link_ids(parent_link_names);
    std::vector<size_t> collision_ids;
    for(size_t i = 0; i < parent_ids.size(); ++i) {
      auto name = "sphere" + std::to_string(i);
      kin_->add_new_link(name, parent_ids[i], 
          {sphere_centers[i].x(), sphere_centers[i].y(), sphere_centers[i].z()},
          {0.0, 0.0, 0.0});
      collision_ids.push_back(kin_->get_link_ids({name})[0]);
    }
    collision_ids_ = collision_ids;
  }

    bool is_valid(const std::vector<double>& q) {
      kin_->set_joint_angles(control_joint_ids_, q);

      tinyfk::Transform pose;
      for (size_t i = 0; i < num_spheres(); ++i) {
        double radius = sphere_radii_[i];
        kin_->get_link_pose(collision_ids_[i], pose);
        Eigen::Vector3d center(pose.position.x, pose.position.y, pose.position.z);
        if(!sdf_->is_outside(center, radius)) {
          return false;
        }
      }
      return true;
    }

    inline size_t num_spheres() const {
      return parent_link_ids_.size();
    }

  private:
    std::vector<size_t> collision_ids_;
    std::shared_ptr<tinyfk::KinematicModel> kin_;
    std::vector<size_t> control_joint_ids_;
    std::vector<size_t> parent_link_ids_;
    std::vector<Eigen::Vector3d> sphere_centers_;
    std::vector<double> sphere_radii_;
    primitive_sdf::SDFBase::Ptr sdf_;
};

PYBIND11_MODULE(_fused, m) {
  m.doc() = "experimental";
  m.def("fuck", [](std::shared_ptr<tinyfk::KinematicModel> p){ });
  py::class_<FusedSpheresCollisionChecker>(m, "FusedSpheresCollisionChecker")
      .def(py::init<const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<Eigen::Vector3d>&,
                    const std::vector<double>&,
                    primitive_sdf::SDFBase::Ptr>())
      .def("is_valid", &FusedSpheresCollisionChecker::is_valid);
  py::class_<Pose>(m, "Pose", py::module_local()).def(
      py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>());
  py::class_<SDFBase, SDFBase::Ptr>(
      m, "SDFBase", py::module_local());  // user is not supposed to instantiate this class. This to
                      // tell pybind that this is a base class
  py::class_<UnionSDF, UnionSDF::Ptr, SDFBase>(m, "UnionSDF", py::module_local())
      .def(py::init<std::vector<SDFBase::Ptr>>())
      .def("evaluate_batch", &UnionSDF::evaluate_batch)
      .def("evaluate", &UnionSDF::evaluate);
  py::class_<BoxSDF, BoxSDF::Ptr, SDFBase>(m, "BoxSDF", py::module_local())
      .def(py::init<const Eigen::Vector3d&, const Pose&>())
      .def("evaluate_batch", &BoxSDF::evaluate_batch)
      .def("evaluate", &BoxSDF::evaluate);
  py::class_<CylinderSDF, CylinderSDF::Ptr, SDFBase>(m, "CylinderSDF", py::module_local())
      .def(py::init<double, double, const Pose&>())
      .def("evaluate_batch", &CylinderSDF::evaluate_batch)
      .def("evaluate", &CylinderSDF::evaluate);
  py::class_<SphereSDF, SphereSDF::Ptr, SDFBase>(m, "SphereSDF", py::module_local())
      .def(py::init<double, const Pose&>())
      .def("evaluate_batch", &SphereSDF::evaluate_batch)
      .def("evaluate", &SphereSDF::evaluate);
}
