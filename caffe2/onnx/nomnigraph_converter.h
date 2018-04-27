#pragma once

#include "onnx/onnx_pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace onnx {
using ::ONNX_NAMESPACE::GraphProto;
}

namespace caffe2 {

nom::repr::NNModule convertToNNModule(onnx::GraphProto &onnxGraph);
onnx::GraphProto convertToONNXGraphProto(nom::repr::NNModule&);

} // namespace caffe2
