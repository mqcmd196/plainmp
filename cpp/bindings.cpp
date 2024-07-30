#include <pybind11/pybind11.h>
#include "constraint.hpp"
#include "psdf.hpp"
#include "third/tinyfk_binding.hpp"
namespace py = pybind11;

PYBIND11_MODULE(_fused, m) {
  primitive_sdf::bind_primitive_sdf(m);
  cst::bind_collision_constraints(m);
  tinyfk::bind_primitive_sdf(m);
}
