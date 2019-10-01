#include "gtest/gtest.h"
#include "ATen/ATen.h"

TEST(TestCustomSparse, Basic) {
  at::Tensor t = at::randn({2,2});
  auto t_s = t.to_custom_sparse("coo");
  t_s.to_dense();
}
