#include "tinyfk.hpp"
#include "tinyfk_binding.hpp"
#include <memory>

namespace tinyfk {

namespace py = pybind11;

void bind_tinyfk(py::module& m) {

  auto m_tinyfk = m.def_submodule("tinyfk");
  py::class_<tinyfk::KinematicModel, std::shared_ptr<tinyfk::KinematicModel>>(m_tinyfk, "KinematicModel" )
      .def(py::init<std::string &>());
}

}  // namespace tinyfk
