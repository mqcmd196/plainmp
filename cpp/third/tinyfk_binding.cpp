#include "tinyfk.hpp"
#include "tinyfk_binding.hpp"

namespace tinyfk {

namespace py = pybind11;

void bind_tinyfk(py::module& m) {

  auto m_psdf = m.def_submodule("tinyfk");
  py::class_<tinyfk::KinematicModel>(m, "KinematicModel" )
      .def(py::init<std::string &>());
}

}  // namespace tinyfk
