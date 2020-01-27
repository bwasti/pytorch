#pragma once

#include <torch/csrc/jit/ir.h>
#include <pybind11/pybind11.h>

/* `getCustomPasses()` returns a vector of passes that will be executed after
 * differentiation but before any fusion.  This is the de-facto location
 * for compiler backends to insert passes.
 *
 * Static registration of a pass can be done by creating a global
 * `RegisterPass r(Pass)` variable in a compilation unit.
 *
 * pass_manager.h uses a Meyer's singleton
 * to store a vector of `Pass`es, which modify the IR graph in place.
 */

namespace torch {
namespace jit {

// A pass modifies a Graph in place.
using Pass = std::function<void(std::shared_ptr<Graph>&)>;

TORCH_API std::vector<Pass>& getCustomPasses();

struct TORCH_API RegisterPass {
  RegisterPass(Pass p);
};

using InitFunc = std::function<void(pybind11::module&)>;

TORCH_API std::vector<InitFunc>& getCustomInitFuncs();

struct TORCH_API RegisterCustomInit {
  RegisterCustomInit(InitFunc p);
};


} // namespace jit
} // namespace torch
