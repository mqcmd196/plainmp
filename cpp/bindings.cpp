#include <pybind11/pybind11.h>
namespace py = pybind11;


void bind_primitive_sdf(py::module &m);
void bind_collision_constraints(py::module &m);

PYBIND11_MODULE(_fused, m) {
  bind_primitive_sdf(m);
  bind_collision_constraints(m);
}
