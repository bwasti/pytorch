#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

class Profiler2 {
 public:
  Profiler2(std::string name, size_t achieved_flops, size_t peak_flops)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()),
        flops(achieved_flops),
        peak(peak_flops) {}
  ~Profiler2() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    float gflops = float(flops) / (dur.count() * float(1000));
    float peak_gflops = float(peak) / 1e9;
    std::cout << m_name << " : " << dur.count() << "us " << gflops << "GFlops ("
              << 100.f * gflops / peak_gflops << "% of peak)\n";
  }

 private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
  size_t flops;
  size_t peak;
};


class LLVMCodeGenImpl;

class TORCH_API LLVMCodeGen : public CodeGen {
 public:
  explicit LLVMCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& args,
      at::Device device = at::kCPU,
      Dtype dtype = kInt);
  explicit LLVMCodeGen(Stmt* stmt);

  LLVMCodeGen() = delete;
  ~LLVMCodeGen() override;

  TORCH_API void call(const std::vector<CallArg>& args) override;

  template <typename T>
  T value() {
    return value<T>(nullptr);
  }

  template <typename T>
  T value(std::vector<void*>& args) {
    return value<T>(args.data());
  }

  template <typename T>
  T value(void** args) {
    T (*fp)(void**) = (T(*)(void**))getKernelAddress(impl_.get());
    T rv = fp(args);
    constexpr size_t warmup = 1e2;
    constexpr size_t iters = 1e7;
    for (size_t i = 0; i < warmup; ++i) {
      T rv = fp(args);
    }
    {
      Profiler2 p("llvm vec rfactor", 64 * 128 * iters * 1<<6, 32 * 2.5 * 1e9 / 4);
      for (size_t i = 0; i < iters; ++i) {
        T rv = fp(args);
      }
    }
    return rv;
  }

 private:
  void* getKernelAddress(LLVMCodeGenImpl* impl);

  std::unique_ptr<LLVMCodeGenImpl> impl_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM
