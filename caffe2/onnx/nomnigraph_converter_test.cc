#include <gtest/gtest.h>
#include "caffe2/opt/converter.h"
#include "nomnigraph_converter.h"
#include <iostream>

#define ADD_ARG(_op, _name, _type, _val)                                     \
{                                                                            \
  caffe2::Argument *arg = _op->add_arg();                                    \
  arg->set_name(_name);                                                      \
  arg->set_##_type(_val);                                                    \
}

TEST(Nomnigraph, ExportTest) {
  caffe2::NetDef net;
  auto* op = net.add_op();
  op->set_type("Conv");
  op->add_input("X");
  op->add_input("W");
  op->add_input("b");
  op->add_output("Y");
  ADD_ARG(op, "kernel", i, 3);
  ADD_ARG(op, "stride", i, 1);
  ADD_ARG(op, "pad", i, 0);
  ADD_ARG(op, "order", s, "NCHW");
  op = net.add_op();
  op->set_type("Relu");
  op->add_input("Y");
  op->add_output("Y");
  net.add_external_input("X");
  net.add_external_output("Y");
  auto nn = caffe2::convertToNNModule(net);
  auto out = caffe2::convertToONNXGraphProto(nn);
  std::cout << out.DebugString();
}

