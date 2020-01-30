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
#include "torch/csrc/jit/passes/quantization.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/init.h"
#include <ATen/core/LegacyTypeDispatch.h>

#include <unordered_set>

using namespace torch::jit;

void inlineIfBody(Block* body) {
  Node* n = body->owningNode();
  for (auto it = body->nodes().begin(); it != body->nodes().end();) {
    Node* body_node = *it;
    // advance iterator because after body_node is moved its next pointer will
    // be to n
    it++;
    body_node->moveBefore(n);
  }
  for (size_t i = 0; i < n->outputs().size(); ++i) {
    n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
  }
  // NB: destroy the node here, because it might contain side effects, like
  // print
  n->destroy();
}

// removes paths that raise runtime exceptions
// Algorithm:
//   find exceptions and backpropagate
//   their presence to nearest ancestor "If" nodes,
//   mark each "If"
void noExcept(torch::jit::Graph* g) {
  std::vector<Node*> frontier;
  std::vector<Node*> new_frontier;

  // immediate parent map
  // node -> if statement, condition
  std::unordered_map<
    Node*, 
    std::pair<Node*, bool>
    > node_to_if;

  // if_node -> condition to raise
  std::unordered_map<Node*, bool> raising_if;

  for (auto n : g->nodes()) {
    frontier.emplace_back(n);
  }

  while (frontier.size()) {
    new_frontier.clear();
    for (auto n : frontier) {
      // we necessarily traversed parents already
      if (n->kind() == c10::Symbol::fromQualString("prim::RaiseException")) {
        TORCH_CHECK(node_to_if.count(n) != 0);
        auto if_parent = node_to_if[n];
        raising_if[if_parent.first] = if_parent.second;
      }
      if (n->kind() != c10::Symbol::fromQualString("prim::If")) {
        continue;
      }
      TORCH_CHECK(n->blocks().size() == 2, "Can't handle prim::If node with != 2 blocks");
      auto true_block = n->blocks()[0];
      auto false_block = n->blocks()[1];
      for (auto n_ : true_block->nodes()) {
        new_frontier.emplace_back(n_);
        node_to_if[n_] = std::make_pair(n, true);
      }
      for (auto n_ : false_block->nodes()) {
        new_frontier.emplace_back(n_);
        node_to_if[n_] = std::make_pair(n, false);
      }
    }
    frontier = new_frontier;
  }

  for (auto kv : raising_if) {
    auto n = kv.first;
    if (!n->outputs().size()) {
      n->destroy();
      continue;
    }
    TORCH_CHECK(n->outputs().size() == n->blocks()[0]->outputs().size());
    TORCH_CHECK(n->outputs().size() == n->blocks()[1]->outputs().size());
    // kv.second == 1 if raises in true block, coincidentally the index
    // we want (the false block)
    auto new_block = n->blocks()[kv.second];
    inlineIfBody(new_block);
  }
}
//void sparsify(py::object& obj) {//torch::jit::script::Module & mod) {
//  auto opt_mod = script::as_module(obj);
//  TORCH_CHECK(opt_mod.has_value());
//  auto mod = opt_mod.value();
void sparsify(script::Module & mod, std::vector<bool> swap_to_512) {
  mod = freeze_module(mod);
  auto f = mod.get_method("forward");
  Inline(*f.graph());
  auto g = f.graph();
  ConstantPropagation(g);
  Canonicalize(f.graph());
  noExcept(f.graph().get());
  EliminateDeadCode(f.graph());
  c10::impl::FLAGS_disable_variable_dispatch = true;

  //for (auto n: f.graph()->nodes()) {
  //  if (n->kind() == c10::Symbol::fromQualString("aten::batch_norm")) {
  //    auto prev_n = n->inputs()[0]->node();
  //    if (!prev_n->kind() == c10::Symbol::fromQualString("aten::conv2d")) {
  //      std::cerr << "BN cannot be fused in previous node of type " << prev_n->kind().toQualString() << "\n";
  //      continue;
  //    }

  //    if (n->inputs()[0]->uses().size() > 1) {
  //      std::cerr << "BN cannot be fused as input activations are used twice:\n";
  //      for (auto use : n->inputs()[0]->uses()) {
  //        std::cerr << "\t" << *use.user << "\n";
  //      }
  //      continue;
  //    }
  //    //std::cerr << "woo!\n";
  //    auto gt = [](Node* n, size_t index) {
  //      Value* v = n->inputs()[index];
  //      auto opt = torch::jit::toIValue(v);
  //      TORCH_CHECK(opt.has_value());
  //      //std::cerr << *opt->type() << "\n";
  //      return opt.value();//.toTensor();
  //    };
  //    auto weight = gt(n, 1).toTensor();
  //    auto bias = gt(n, 2).toTensor();
  //    auto running_mean = gt(n, 3).toTensor();
  //    auto running_var = gt(n, 4).toTensor();
  //    // 5 training, 6 momentum
  //    auto eps = gt(n, 7).toDouble();
  //    // w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
  //    // b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
  //    //std::cerr << "calc wbn\n";
  //    auto w_bn = at::diag(weight / (at::sqrt(eps + running_var)));
  //    //std::cerr << "calc bbn\n";
  //    auto b_bn = bias - weight * running_mean / at::sqrt(running_var + eps);
  //    auto w_conv = gt(prev_n, 1).toTensor();
  //    auto w_conv_flat = w_conv.view({w_conv.sizes()[0], -1});
  //    auto b_conv = gt(prev_n, 2).toTensor();
  //    //std::cerr << "calc new_w\n";
  //    auto new_w = at::mm(w_bn, w_conv_flat).reshape_as(w_conv);
  //    ////std::cerr << "calc new_b\n";
  //    auto new_b = b_conv + b_bn;
  //    ////std::cerr << "done calc\n";
  //    auto new_w_v = f.graph()->insertConstant(new_w, c10::nullopt, prev_n->scope());
  //    auto new_b_v = f.graph()->insertConstant(new_b, c10::nullopt, prev_n->scope());
  //    new_w_v->node()->moveBefore(prev_n);
  //    new_b_v->node()->moveBefore(prev_n);
  //    //std::cerr << "dont insert\n";
  //    prev_n->inputs()[1]->replaceAllUsesWith(new_w_v);
  //    prev_n->inputs()[2]->replaceAllUsesWith(new_b_v);
  //    //std::cerr << "replacing node \n";
  //    //std::cerr << "n has " << n->outputs().size() << " outputs\n";
  //    for (auto out : n->outputs()) {
  //      //std::cerr << "\t" << out->uses().size() << " uses\n";
  //    }
  //    n->replaceAllUsesWith(prev_n);
  //    f.graph()->lint();
  //    //std::cerr << "done repalce\n";
  //  }
  //}

  size_t count = 0;
  for (auto n: f.graph()->nodes()) {
    if (n->kind() == c10::Symbol::fromQualString("aten::conv2d")) {
      auto kernel = torch::jit::toIValue(n->inputs()[1]);
      if (!kernel.has_value()) {
        std::cerr << "can't handle non-constants weights\n";
        continue;
      }
      auto k = kernel->toTensor().sizes().at(2);
      auto channels = kernel->toTensor().sizes().at(0);
      if (k != 1 && k != 3) {
        std::cerr << "k is " << k << "\n";
        continue;
      }
      auto stride = torch::jit::toIValue(n->inputs()[3]);
      auto padding = torch::jit::toIValue(n->inputs()[4]);
      auto dilation = torch::jit::toIValue(n->inputs()[5]);
      auto groups_ival = torch::jit::toIValue(n->inputs()[6]);
      int64_t groups = 1;
      if (groups_ival.has_value()) {
	groups = groups_ival.value().toInt();
      }
      // TODO ENABLE DW
      if (groups != 1) continue;
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
      if (groups == channels && k == 3) {
        //std::cerr << "groups are fine " << groups << "\n";
      } else if (groups != 1) {
        std::cerr << "Couldn't handle channels "<<channels << "groups " << groups << " " << k << "x" << k << "\n";
        continue; 
      }

      //std::cerr << "Handling conv\n";

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
      bool use_512 = false;
      if (swap_to_512.size() > count) {
        use_512 = swap_to_512[count];
      }
      auto padding_n = g->insertConstant(padding_, c10::nullopt, sparseConv->scope());
      auto stride_n = g->insertConstant(1, c10::nullopt, sparseConv->scope());
      auto group_n = g->insertConstant(groups, c10::nullopt, sparseConv->scope());
      auto use_512_n = g->insertConstant(use_512, c10::nullopt, sparseConv->scope());
      sparseConv->addInput(stride_n);
      sparseConv->addInput(padding_n);
      sparseConv->addInput(group_n);
      sparseConv->addInput(use_512_n);
      stride_n->node()->moveBefore(sparseConv);
      padding_n->node()->moveBefore(sparseConv);
      group_n->node()->moveBefore(sparseConv);
      use_512_n->node()->moveBefore(sparseConv);
      n->replaceAllUsesWith(sparseConv);
      g->lint();
      count++;
    }
  }
  std::cerr << "Found " << count << " convs to swap\n";
  EliminateDeadCode(f.graph());
  //std::cerr << *f.graph() << "\n";
}

namespace torch {
namespace jit {

//TORCH_API std::vector<InitFunc>& getCustomInitFuncs();
//struct TORCH_API RegisterCustomInit {
//  RegisterCustomInit(InitFunc p);
//};
std::unordered_map<std::string, InitFunc>& getCustomInitFuncs() {
  static std::unordered_map<std::string, InitFunc> funcs;
  return funcs;
}

RegisterCustomInit::RegisterCustomInit(std::string name, InitFunc f) {
  getCustomInitFuncs()[name] = std::move(f);
}

static RegisterCustomInit initF("sparsify", &sparsify);


std::vector<Pass>& getCustomPasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPass::RegisterPass(Pass p) {
  getCustomPasses().emplace_back(std::move(p));
}

} // namespace jit
} // namespace torch
