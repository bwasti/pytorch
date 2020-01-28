#include "torch/csrc/jit/tensorexpr/ir_mutator.h"

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static Expr mutate_binary_op(
    const BinaryOpNode<Op>* v,
    IRMutator* mutator,
    bool option = false) {
  Expr lhs = v->lhs();
  Expr rhs = v->rhs();
  Expr lhs_new = lhs.accept_mutator(mutator);
  Expr rhs_new = rhs.accept_mutator(mutator);
  if (same_node(lhs, lhs_new) && same_node(rhs, rhs_new)) {
    return Expr(v);
  }
  IRNodeType expr_type = v->expr_type();
  switch (expr_type) {
    case IRNodeType::kAdd:
      return Add::make(lhs_new, rhs_new);
    case IRNodeType::kSub:
      return Sub::make(lhs_new, rhs_new);
    case IRNodeType::kMul:
      return Mul::make(lhs_new, rhs_new);
    case IRNodeType::kDiv:
      return Div::make(lhs_new, rhs_new);
    case IRNodeType::kMax:
      return Max::make(lhs_new, rhs_new, option);
    case IRNodeType::kMin:
      return Min::make(lhs_new, rhs_new, option);
    default:
      LOG(FATAL) << "unsupported expr_type" << static_cast<int>(expr_type);
      return Expr();
  }
}

Expr IRMutator::mutate(const Add* v) {
  return mutate_binary_op(v, this);
}

Expr IRMutator::mutate(const Sub* v) {
  return mutate_binary_op(v, this);
}

Expr IRMutator::mutate(const Mul* v) {
  return mutate_binary_op(v, this);
}

Expr IRMutator::mutate(const Div* v) {
  return mutate_binary_op(v, this);
}

Expr IRMutator::mutate(const Max* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

Expr IRMutator::mutate(const Min* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

Expr IRMutator::mutate(const CompareSelect* v) {
  Expr lhs = v->lhs();
  Expr rhs = v->rhs();
  Expr lhs_new = lhs.accept_mutator(this);
  Expr rhs_new = rhs.accept_mutator(this);
  if (same_node(lhs, lhs_new) && same_node(rhs, rhs_new)) {
    return Expr(v);
  }
  return CompareSelect::make(lhs_new, rhs_new, v->compare_select_op());
}

Expr IRMutator::mutate(const IntImm* v) {
  return Expr(v);
}

Expr IRMutator::mutate(const FloatImm* v) {
  return Expr(v);
}

Expr IRMutator::mutate(const Cast* v) {
  Expr src_value = v->src_value();
  Expr src_value_new = src_value.accept_mutator(this);
  if (same_node(src_value_new, v->src_value())) {
    return Expr(v);
  }
  return Cast::make(v->dtype(), src_value_new);
}

Expr IRMutator::mutate(const Variable* v) {
  return Expr(v);
}

Expr IRMutator::mutate(const Let* v) {
  Expr var = v->var();
  Expr value = v->value();
  Expr body = v->body();
  Expr var_new = var.accept_mutator(this);
  Expr value_new = value.accept_mutator(this);
  Expr body_new = body.accept_mutator(this);
  if (same_node(var, var_new) && same_node(value, value_new) &&
      same_node(body, body_new)) {
    return Expr(v);
  }
  return Let::make(var_new, value_new, body_new);
}

Expr IRMutator::mutate(const Ramp* v) {
  Expr base = v->base();
  Expr stride = v->stride();
  Expr base_new = base.accept_mutator(this);
  Expr stride_new = stride.accept_mutator(this);
  if (same_node(base, base_new) && same_node(stride, stride_new)) {
    return Expr(v);
  }
  return Ramp::make(base_new, stride_new, v->lanes());
}

Expr IRMutator::mutate(const Load* v) {
  Dtype dtype = v->dtype();
  Var base_handle = v->base_handle();
  Expr index = v->index();
  Expr mask = v->mask();
  Expr base_handle_expr = base_handle.accept_mutator(this);
  Var base_handle_new = Var(base_handle_expr.AsNode<Variable>());
  Expr index_new = index.accept_mutator(this);
  Expr mask_new = mask.accept_mutator(this);
  if (same_node(base_handle, base_handle_new) && same_node(index, index_new) &&
      same_node(mask, mask_new)) {
    return Expr(v);
  }
  return Load::make(dtype, base_handle_new, index_new, mask_new);
}

Expr IRMutator::mutate(const Broadcast* v) {
  Expr value = v->value();
  int lanes = v->lanes();
  Expr value_new = value.accept_mutator(this);
  if (same_node(value, value_new)) {
    return Expr(v);
  }
  return Broadcast::make(value_new, lanes);
}

Expr IRMutator::mutate(const Intrinsics* v) {
  const BaseCallNode* base = v;
  return this->mutate(base);
}

Expr IRMutator::mutate(const FunctionCall* v) {
  const BaseCallNode* base = v;
  return this->mutate(base);
}

Expr IRMutator::mutate(const BaseCallNode* v) {
  std::vector<Expr> params(v->nparams());
  bool any_change = false;
  for (int i = 0; i < v->nparams(); i++) {
    Expr value = v->param(i);
    Expr value_new = value.accept_mutator(this);
    if (!same_node(value, value_new)) {
      any_change = true;
    }
    params[i] = std::move(value_new);
  }
  if (!any_change) {
    return Expr(v);
  }
  return v->DefaultMutator(params);
}

Stmt IRMutator::mutate(const For* v) {
  Var var = v->var();
  Expr start = v->start();
  Expr stop = v->stop();
  Stmt body = v->body();
  Expr var_new_expr = var.accept_mutator(this);
  Var var_new = Var(var_new_expr.AsNode<Variable>());
  Expr start_new = start.accept_mutator(this);
  Expr stop_new = stop.accept_mutator(this);
  Stmt body_new = body.accept_mutator(this);
  if (same_node(var, var_new) && same_node(start, start_new) &&
      same_node(stop, stop_new) && same_node(body, body_new)) {
    return Stmt(v);
  }
  return For::make(var_new, start_new, stop_new, body_new);
}

Stmt IRMutator::mutate(const Block* v) {
  bool any_change = false;
  std::vector<Stmt> stmts;
  for (int i = 0; i < v->nstmts(); i++) {
    Stmt stmt = v->stmt(i);
    Stmt stmt_new = stmt.accept_mutator(this);
    if (!same_node(stmt, stmt_new)) {
      any_change = true;
    }
    stmts.push_back(stmt_new);
  }
  if (!any_change) {
    return Stmt(v);
  }
  return Block::make(stmts);
}

Stmt IRMutator::mutate(const Store* v) {
  Var base_handle = v->base_handle();
  Expr index = v->index();
  Expr value = v->value();
  Expr mask = v->mask();
  Expr base_handle_expr = base_handle.accept_mutator(this);
  Var base_handle_new = Var(base_handle_expr.AsNode<Variable>());
  Expr index_new = index.accept_mutator(this);
  Expr value_new = value.accept_mutator(this);
  Expr mask_new = mask.accept_mutator(this);
  if (same_node(base_handle, base_handle_new) && same_node(index, index_new) &&
      same_node(value, value_new) && same_node(mask, mask_new)) {
    return Stmt(v);
  }
  return Store::make(base_handle_new, index_new, value_new, mask_new);
}

Stmt IRMutator::mutate(const Allocate* v) {
  Var buffer_var_old = v->buffer_var();
  Var buffer_var_new =
      Var(buffer_var_old.accept_mutator(this).AsNode<Variable>());
  bool any_change = same_node(buffer_var_new, buffer_var_old);

  std::vector<Expr> dims_old = v->dims();
  std::vector<Expr> dims_new(dims_old.size());
  for (size_t i = 0; i < dims_old.size(); i++) {
    dims_new[i] = dims_old[i].accept_mutator(this);
    any_change |= same_node(dims_new[i], dims_old[i]);
  }

  if (!any_change) {
    return Stmt(v);
  }

  return Allocate::make(buffer_var_new, v->dtype(), dims_new);
}

Stmt IRMutator::mutate(const Free* v) {
  Var buffer_var_old = v->buffer_var();
  Var buffer_var_new =
      Var(buffer_var_old.accept_mutator(this).AsNode<Variable>());
  if (same_node(buffer_var_new, buffer_var_old)) {
    return Stmt(v);
  }

  return Free::make(buffer_var_new);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
