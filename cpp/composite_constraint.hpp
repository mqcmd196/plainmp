#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <memory>
#include <tinyfk.hpp>
#include <utility>
#include "constraint.hpp"
#include "primitive_sdf.hpp"

namespace cst {
template <typename T>
class CompositeConstraintBase {
 public:
  using Ptr = std::shared_ptr<CompositeConstraintBase>;
  CompositeConstraintBase(std::vector<T> constraints)
      : constraints_(constraints) {}

  void update_kintree(const std::vector<double>& q) {
    constraints_.front()->update_kintree(q);
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate() {
    size_t dim = this->cst_dim();
    Eigen::VectorXd vals(dim);
    Eigen::MatrixXd jac(dim, q_dim());
    size_t head = 0;
    for (const auto& cst : constraints_) {
      size_t dim_local = cst->cst_dim();
      auto [vals_sub, jac_sub] = cst->evaluate();
      vals.segment(head, dim_local) = vals_sub;
      jac.block(head, 0, dim_local, q_dim()) = jac_sub;
      head += dim_local;
    }
    return {vals, jac};
  }

  size_t q_dim() const { return constraints_.front()->q_dim(); }

  size_t cst_dim() const {
    return std::accumulate(
        constraints_.begin(), constraints_.end(), 0,
        [](size_t sum, const T& cst) { return sum + cst->cst_dim(); });
  }
  std::vector<T> constraints_;
};

class EqCompositeCst : public CompositeConstraintBase<EqConstraintBase::Ptr> {
 public:
  using Ptr = std::shared_ptr<EqCompositeCst>;
  using CompositeConstraintBase::CompositeConstraintBase;
  size_t cst_dim() const;
  bool is_equality() const { return true; }
};

class IneqCompositeCst
    : public CompositeConstraintBase<IneqConstraintBase::Ptr> {
 public:
  using Ptr = std::shared_ptr<IneqCompositeCst>;
  using CompositeConstraintBase::CompositeConstraintBase;
  bool is_valid() const {
    for (const auto& cst : constraints_) {
      if (!cst->is_valid())
        return false;
    }
    return true;
  }
  size_t cst_dim() const;
  bool is_equality() const { return false; }
};
}  // namespace cst
