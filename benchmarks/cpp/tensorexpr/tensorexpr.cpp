#include <benchmark/benchmark.h>
#include "torch/torch.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace te = torch::jit::tensorexpr;

class Gemm : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) {
    M = state.range(0);
    N = state.range(1);
    K = state.range(2);
    A = torch::randn({M, K});
    B = torch::randn({K, N});
    C = torch::mm(A, B);
  }

  void TearDown(benchmark::State& state) {
    state.counters["GFLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * M * N * K,
                         benchmark::Counter::kIsRate);
  }

  int M;
  int N;
  int K;
  at::Tensor A;
  at::Tensor B;
  at::Tensor C;
};

BENCHMARK_DEFINE_F(Gemm, Torch)(benchmark::State& state) {
  for (auto _ : state) {
    torch::mm_out(C, A, B);
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprNoopt)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m, const te::ExprHandle& n, const te::ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  te::LoopNest loop({CT});
  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile32x32)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m, const te::ExprHandle& n, const te::ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* m = loops[0];
    te::For* mo;
    te::For* mi;
    loop.splitWithMask(m, 32, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* n = loops[2];
    te::For* no;
    te::For* ni;
    loop.splitWithMask(n, 32, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[1];
    te::For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[3];
    te::For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[2];
    te::For* k = loops[3];
    loop.reorderAxis(mi, k);
  }

  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

/*
Trying to get this layout:

for n₀ in 2
 for m₀ in 32
  for k₀ in 8
   for k₁ in 16   # unroll, so we have 64 FMAs total
    for m₁ in 4   # 4 sets of vector regs, 16 regs total (4 non-data dependent FMAs)
     for n₁ in 64 # 4 vector regs
      fma(c[m₀, m₁, n₀, n₁], a[m₀, m₁, k₀, k₁], b[k₀, k₁, n₀, n₁])

an even better version below but I don't know how to mess with tail logic

for n₀ in 2
 for m₀ in 25
  for k₀ in 8
   for k₁ in 16
    for m₁ in 5 # tail loop below
     for n₁ in 64
      fma(c[m₀, m₁, n₀, n₁], a[m₀, m₁, k₀, k₁], b[k₀, k₁, n₀, n₁])
 for k₀ in 8
  for k₁ in 16
   for m₁ in 3
    for n₁ in 64
     fma(c[m₀, m₁, n₀, n₁], a[m₀, m₁, k₀, k₁], b[k₀, k₁, n₀, n₁])
*/
BENCHMARK_DEFINE_F(Gemm, TensorExprOptAVX512)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m, const te::ExprHandle& n, const te::ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* m = loops[0];
    te::For* mo;
    te::For* mi;
    loop.splitWithMask(m, 4, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* n = loops[2];
    te::For* no;
    te::For* ni;
    loop.splitWithMask(n, 64, &no, &ni);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* k = loops[4];
    te::For* ko;
    te::For* ki;
    loop.splitWithMask(k, 16, &ko, &ki);
  }
  std::cout << "before reorder: " << *loop.root_stmt() << "\n";
  // mo, mi, no, *ni, ko, *ki ->
  // mo, mi, no, *ki, ko, *ni ->
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[3];
    te::For* ki = loops[5];
    loop.reorderAxis(ni, ki);
  }
  std::cout << "reorder 0: " << *loop.root_stmt() << "\n";
  // mo, *mi, no, ki, *ko, ni ->
  // mo, *ko, no, ki, *mi, ni ->
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[1];
    te::For* ko = loops[4];
    loop.reorderAxis(mi, ko);
  }
  std::cout << "reorder 1: " << *loop.root_stmt() << "\n";
  // mo, *ko, *no, ki, mi, ni ->
  // mo, *no, *ko, ki, mi, ni ->
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ko = loops[1];
    te::For* no = loops[2];
    loop.reorderAxis(ko, no);
  }
  std::cout << "reorder 2: " << *loop.root_stmt() << "\n";
  // mo, no, ko, ki, mi, *ni ->
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[5];
    loop.vectorize(ni);
  }
  std::cout << "vectorize 0: " << *loop.root_stmt() << "\n";
  // mo, no, ko, ki, *mi, ni ->
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[4];
    //loop.vectorize(mi);
  }
  std::cout << "vectorize 1: " << *loop.root_stmt() << "\n";
  // mo, no, ko, *ki, mi, ni ->
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ki = loops[3];
    te::Stmt* u = nullptr;
    //loop.unroll(ki, &u);
  }
  std::cout << "final:" << *loop.root_stmt() << "\n";

  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_REGISTER_F(Gemm, Torch)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprNoopt)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprTile32x32)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprOptAVX512)->Args({128, 128, 128});

BENCHMARK_MAIN();
