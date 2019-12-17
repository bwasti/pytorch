#include <stdexcept>

#include "expr.h"
#include "ir.h"

#include <gtest/gtest.h>
#include "test_utils.h"

namespace nnc {

TEST(ExprTest, BasicValueTest) {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleExprEvaluator eval;
  c.accept(&eval);
  EXPECT_EQ(eval.value().as<int>(), 5);
}

TEST(ExprTest, BasicValueTest02) {
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);
  SimpleExprEvaluator eval;
  f.accept(&eval);
  EXPECT_EQ(eval.value().as<float>(), -4.0f);
}

TEST(ExprTest, LetTest01) {
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  SimpleExprEvaluator eval;
  result.accept(&eval);
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4));
}

TEST(ExprTest, LetTest02) {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  SimpleExprEvaluator eval;
  e2.accept(&eval);
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4 * 6));
}

TEST(ExprTest, Tensor01) {
  Tensor tensor = Compute({Expr(3), Expr(4)}, {"x", "y"},
                          [](const Var& x, const Var& y) {
                            return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
                          });
  std::vector<float> result;
  SimpleTensorEvaluator<float> tensor_eval;
  tensor_eval.evaluate(tensor, &result);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float reference_v = 1 + i * i + j * j;
      int index = i * 4 + j;
      EXPECT_EQ(result[index], reference_v);
    }
  }
}

TEST(ExprTest, VectorAdd01) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  /*
  Build the following:
    for (int index = 0; index < kVectorSize; index++) {
      store(c_buf, ramp(index * 8, 1, 8),
            load(a_buf, ramp(index * 8, 1, 8) +
            load(b_buf, ramp(index * 8, 1, 8))))
    }
  */
  Var index = Var("index", kInt32);
  Expr load_a = Load::make(a_buf, Ramp::make(index * kVectorSize, 1, kVectorSize),
                           Broadcast::make(1, kVectorSize));
  Expr load_b = Load::make(b_buf, Ramp::make(index * kVectorSize, 1, kVectorSize),
                           Broadcast::make(1, kVectorSize));
  Expr value = load_a + load_b;
  Stmt store_c = Store::make(c_buf, Ramp::make(index * kVectorSize, 1, kVectorSize), value,
                             Broadcast::make(1, kVectorSize));
  Stmt stmt = For::make(index, 0, kVectorSize, store_c);

  EXPECT_EQ(load_a.dtype(), Dtype(kFloat32, kVectorSize));
  EXPECT_EQ(load_b.dtype(), Dtype(kFloat32, kVectorSize));
  EXPECT_EQ(value.dtype(), Dtype(kFloat32, kVectorSize));
}

}  // namespace nnc
