#include <pybind11/pybind11.h>
#include "tinyfk.hpp"

namespace tinyfk {

namespace py = pybind11;

void bind_primitive_sdf(py::module& m) {
  auto m_psdf = m.def_submodule("tinyfk");
  py::class_<tinyfk::KinematicModel>(m, "KinematicModel")
      .def(py::init<std::string&>());
}

}  // namespace tinyfk
