#include "fusion.h"
#include "nomnigraph/Converters/Caffe2.h"

namespace caffe2 {
namespace opt {

using namespace nom;

template <typename OperationT, typename ActivationT, typename FusedT>
bool fusionHelper(std::string fusedName, repr::NNGraph* g) {
  for (auto node_pair : repr::nn::dataIterator<OperationT>(*g)) {
    repr::NNGraph::NodeRef node;
    OperationT* operation;
    std::tie(operation, node) = node_pair;

    // Single output check (intrinsic to a operation, but we double check)
    auto outputs = repr::nn::getOutputs(node);
    if (outputs.size() != 1) {
      continue;
    }
    auto tensorNode = outputs.front();

    // Single user check.
    auto consumers = repr::nn::getConsumers(tensorNode);
    if (consumers.size() != 1) {
      continue;
    }

    // Followed by Activation check.
    auto* nextNode = consumers.front();
    if (!repr::nn::is<ActivationT>(nextNode)) {
      continue;
    }

    // Naming for operationenience
    auto* operationNode = node;
    auto* reluNode = nextNode;

    // Create our Operation + Activation and annotate it by modifying the
    // original Operation
    auto* fusedNode = g->createNode(util::make_unique<FusedT>(*operation));
    auto fused = repr::nn::get<FusedT>(fusedNode);
    fused->setAnnotation(util::make_unique<repr::Annotation>());
    auto annotation = fused->getMutableAnnotation();

    // Modification of the original Fuseable
    auto oldAnnotation = operation->getAnnotation();
    auto operationOp =
        reinterpret_cast<caffe2::OperatorDef*>(oldAnnotation->getSaved());
    operationOp->set_type(fusedName);
    annotation->setSaved(operationOp);

    for (const auto input : repr::nn::getInputs(operationNode)) {
      g->createEdge(input, fusedNode);
    }
    for (const auto output : repr::nn::getOutputs(operationNode)) {
      g->createEdge(fusedNode, output);
    }

    g->deleteNode(operationNode);
    g->deleteNode(tensorNode);
    g->deleteNode(reluNode);

    return true;
  }
  return false;
}

caffe2::NetDef fuseConvRelu(caffe2::NetDef net) {
  auto nn = nom::converters::convertFromCaffe2Proto(net);
  while (fusionHelper<repr::Conv, repr::Relu, repr::ConvRelu>(
      "ConvRelu", &nn.dataFlow)) {
  }
  return nom::converters::convertToCaffe2Proto(nn);
}

caffe2::NetDef fuseAveragePoolRelu(caffe2::NetDef net) {
  auto nn = nom::converters::convertFromCaffe2Proto(net);
  while (fusionHelper<repr::AveragePool, repr::Relu, repr::AveragePoolRelu>(
      "AveragePoolRelu", &nn.dataFlow)) {
  }
  return nom::converters::convertToCaffe2Proto(nn);
}

caffe2::NetDef fuseMaxPoolRelu(caffe2::NetDef net) {
  auto nn = nom::converters::convertFromCaffe2Proto(net);
  while (fusionHelper<repr::MaxPool, repr::Relu, repr::MaxPoolRelu>(
      "MaxPoolRelu", &nn.dataFlow)) {
  }
  return nom::converters::convertToCaffe2Proto(nn);
}

caffe2::NetDef fuseSumRelu(caffe2::NetDef net) {
  auto nn = nom::converters::convertFromCaffe2Proto(net);
  while (fusionHelper<repr::Sum, repr::Relu, repr::SumRelu>(
      "SumRelu", &nn.dataFlow)) {
  }
  return nom::converters::convertToCaffe2Proto(nn);
}

} // namespace opt
} // namespace caffe2
