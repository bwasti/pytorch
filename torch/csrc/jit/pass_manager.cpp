#include <torch/csrc/jit/pass_manager.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/pybind.h>
#include "torch/csrc/jit/subgraph_matcher.h"
#include "torch/csrc/jit/irparser.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/inliner.h"
#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/canonicalize.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/script/module_python.h"
#include "torch/csrc/jit/pybind_utils.h"
#include "torch/csrc/jit/init.h"
#include <ATen/core/LegacyTypeDispatch.h>

#include <unordered_set>

using namespace torch::jit;


//void sparsify(py::object& obj) {//torch::jit::script::Module & mod) {
//  auto opt_mod = script::as_module(obj);
//  TORCH_CHECK(opt_mod.has_value());
//  auto mod = opt_mod.value();
void sparsify(script::Module & mod) {
  mod = freeze_module(mod);
  auto f = mod.get_method("forward");
  Inline(*f.graph());
  auto g = f.graph();
  ConstantPropagation(g);
  Canonicalize(f.graph());
  c10::impl::FLAGS_disable_variable_dispatch = true;

  for (auto n: f.graph()->nodes()) {
    if (n->kind() == c10::Symbol::fromQualString("aten::conv2d")) {
      auto kernel = torch::jit::toIValue(n->inputs()[1]);
      if (!kernel.has_value()) {
        std::cerr << "can't handle non-constants weights\n";
        continue;
      }
      auto k = kernel->toTensor().sizes().at(2);
      if (k != 1 && k != 3) {
        std::cerr << "k is " << k << "\n";
        continue;
      }
      auto stride = torch::jit::toIValue(n->inputs()[3]);
      auto padding = torch::jit::toIValue(n->inputs()[4]);
      auto dilation = torch::jit::toIValue(n->inputs()[5]);
      auto groups = torch::jit::toIValue(n->inputs()[6]);
      auto only_x = [&](c10::optional<IValue> v, int x, std::string name) {
        if (v.has_value()) {
          if (v->isTuple()) {
            for (auto val : v->toTuple()->elements()) {
              if (val.toInt() != x) {
                std::cerr << "Cannot handle " << name << " " << *v << "\n";
                return false;
              }
            }
          } else if (v->isIntList()) {
			    auto l = v->toIntList();
            for (auto val : l.vec()) {
              if (val != x) {
                std::cerr << "Cannot handle " << name << " " << *v << "\n";
                return false;
              }
            }
          //} else if (v->isList()) {
          //  for (auto val : v->toList()) {
          //    if (IValue(val).toInt() != x) {
          //      std::cerr << "Cannot handle " << name << " " << *v << "\n";
          //      return false;
          //    }
          //  }
          } else if (v->isInt()) {
            if (v->toInt() != x) {
              std::cerr << "Cannot handle "<<name<<" " << *v << "\n";
              return false;
            }
          } else {
            std::cerr << name << " is not an int, list or tuple " << *v << "\n";
            return false;
          }
        } else {
          std::cerr << "Couldn't find " << name << ", assuming valid.  Node: " << *n << "\n";
          //return false;
        }
        return true;
      };
      if (!only_x(stride, 1, "stride")) { continue; }

      if (k == 3) {
        if (!only_x(padding, 1, "padding")) { continue; }
      }
      if (k == 1) {
        if (!only_x(padding, 0, "padding")) { continue; }
      }
      //if (k == 1 && !only_x(padding, 0, "padding")) { continue; }

      if (!only_x(dilation, 1, "dilation")) { continue; }
      if (!only_x(groups, 1, "groups")) { continue; }

      std::cerr << "Handling conv\n";

      TORCH_CHECK(n->outputs().size() == 1);
      auto g = n->owningGraph();
      Node* sparseConv = g->create(
        c10::Symbol::fromQualString("icml::conv"), 1);
      sparseConv->setScope(n->scope());
      sparseConv->insertBefore(n);
      sparseConv->addInput(n->inputs().at(0));
      sparseConv->addInput(n->inputs().at(1));
      sparseConv->addInput(n->inputs().at(2)); // bias
      auto padding_ = 1;
      if (k == 1) {
	      padding_=0;
      }
      auto padding_n = g->insertConstant(padding_, c10::nullopt, sparseConv->scope());
      auto stride_n = g->insertConstant(1, c10::nullopt, sparseConv->scope());
      sparseConv->addInput(stride_n);
      sparseConv->addInput(padding_n);
      stride_n->node()->moveBefore(sparseConv);
      padding_n->node()->moveBefore(sparseConv);
      n->replaceAllUsesWith(sparseConv);
      g->lint();
    }
  }
  EliminateDeadCode(f.graph());
}

static RegisterCustomInit r([](py::module& m) {
  m.def("sparsify", &sparsify);
});

namespace torch {
namespace jit {

std::vector<InitFunc>& getCustomInitFuncs() {
  static std::vector<InitFunc> funcs;
  return funcs;
}

RegisterCustomInit::RegisterCustomInit(InitFunc f) {
  getCustomInitFuncs().emplace_back(std::move(f));
}


std::vector<Pass>& getCustomPasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPass::RegisterPass(Pass p) {
  getCustomPasses().emplace_back(std::move(p));
}

} // namespace jit
} // namespace torch
