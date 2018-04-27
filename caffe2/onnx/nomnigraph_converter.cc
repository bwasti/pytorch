#include "nomnigraph_converter.h"
#include "onnx_exporter.h"
#include "caffe2/opt/converter.h"
#include "nomnigraph/Graph/Algorithms.h"

namespace caffe2 {

using namespace nom;

repr::NNModule convertToNNModule(onnx::GraphProto &onnxGraph) {
  return repr::NNModule();
}

void handleCaffe2BackedNode(repr::NNGraph::NodeRef node, onnx::GraphProto& graphProto) {
  auto nnOp = repr::nn::get<repr::NeuralNetOperator>(node);
  auto* annotation = dyn_cast<Caffe2Annotation>(nnOp->getAnnotation());
  onnx::OnnxExporter onnxExp;
  auto r = onnxExp.Caffe2OpToOnnxNodes(*annotation->getOperatorDef(), {});
  for (auto n : r.first) {
    graphProto.add_node()->CopyFrom(n);
  }
  assert(r.second.size() == 0);
}

onnx::GraphProto convertToONNXGraphProto(repr::NNModule& m) {
  auto graphProto = onnx::GraphProto();

  repr::nn::coalesceInsertedDataDependencies(&m);

  auto sccs = nom::algorithm::tarjans(&m.controlFlow);
  std::reverse(sccs.begin(), sccs.end());

  for (const auto &bbNodeScc : sccs) {
    // If our SCC has more than 1 node there is a loop
    if (bbNodeScc.Nodes.size() > 1) {
      assert(0 && "Loops not yet supported");
    }
    auto bbNode = *bbNodeScc.Nodes.begin();
    if (bbNode->getOutEdges().size() > 1) {
      assert(0 && "Branching not yet supported.");
    }

    auto bb = bbNode->data().get();
    for (const auto &instrNode : bb->getInstructions()) {

      auto nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
      auto* annotation = nnOp->getAnnotation();
      if (isa<Caffe2Annotation>(annotation)) {
        handleCaffe2BackedNode(instrNode, graphProto);
      } else {
        assert(0 && "Not yet supported.");
      }
    }

  }

  return graphProto;
}

} // namespace caffe2
