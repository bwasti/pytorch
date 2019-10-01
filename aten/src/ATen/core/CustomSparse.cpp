#include <c10/core/TensorImpl.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ATenDispatch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

namespace at {

class CustomSparseImpl : public TensorImpl {
  size_t layout_;
  std::shared_ptr<void> storage_;
 public:
  size_t layout() {
    return layout_;
  }
  CustomSparseImpl(
    size_t layout, std::shared_ptr<void> storage, TensorTypeSet ts,
    caffe2::TypeMeta dtype, c10::Device device) :
    TensorImpl(ts.add(TensorTypeId::CustomSparseTensorId),
        dtype,
        device) {
        layout_ = layout;
        storage_ = storage;
        }
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<CustomSparseImpl>(
      layout_,
      storage_,
      type_set(),
      dtype(),
      device_type()
    );
    return impl;
  }
};

void sparseDispatch(const char* schema, torch::jit::Stack* stack);

struct RegisterCustomSparse {
  RegisterCustomSparse() {
    globalATenDispatch().registerFallbackBoxedOp(
        c10::TensorTypeId::CustomSparseTensorId, &sparseDispatch);
  }
};

static RegisterCustomSparse r;

using Impl = void(const char* schema, torch::jit::Stack*);

// layout_name -> layout
static std::unordered_map<std::string, size_t> layout_name_map;
// layout -> (schema -> impl)
static std::unordered_map<size_t, std::unordered_map<std::string, Impl*>> layout_impls;

namespace native {


at::Tensor to_custom_sparse(const Tensor& self, std::string type) {
  TORCH_CHECK(layout_name_map.find(type) != layout_name_map.end());
  size_t layout = layout_name_map[type];
  auto& m = layout_impls[layout];
  auto impl = m["to_sparse"];
  const char* schema = "aten::to_sparse(Tensor self) -> Tensor";
  torch::jit::Stack s;
  torch::jit::push(s, self);
  impl(schema, &s);
  auto tensor = torch::jit::pop(s).toTensor();
  return tensor;
}

} // namespace native

void sparseDispatch(const char* schema, torch::jit::Stack* stack) {
  size_t layout = -1; // meaningless, must be overwritten

  for (const auto& iv : *stack) {
    if (iv.isTensor()) {
      auto t = iv.toTensor();
      auto cst = static_cast<CustomSparseImpl*>(t.unsafeGetTensorImpl());
      layout = cst->layout();
      break;
    }
  }
  TORCH_CHECK(layout_impls.find(layout) != layout_impls.end());

  auto& m = layout_impls[layout];
  auto fs = torch::jit::parseSchema(schema);
  auto impl = m[fs.name()];
  impl(schema, stack);
}

struct RegSparseImpl {
  RegSparseImpl(std::string name, std::string method, Impl* impl) {
    if (layout_name_map.find(name) == layout_name_map.end()) {
      layout_name_map[name] = layout_name_map.size();
    }
    auto layout = layout_name_map[name];
    auto& m = layout_impls[layout];
    TORCH_CHECK(m.find(method) == m.end());
    m[method] = impl;
  }
};

#define REGISTER_SPARSE(name, method, impls) \
  static RegSparseImpl implementations_##name(#name, method, impls);
#define GET_LAYOUT(name) \
  layout_name_map[name];

struct COO {
  at::Tensor c;
};

void to_coo(const char*, torch::jit::Stack* stack) {
  auto self = torch::jit::pop(*stack).toTensor();
  auto layout = GET_LAYOUT("coo");
  std::shared_ptr<void> s = std::static_pointer_cast<void>(std::make_shared<COO>());
  auto tensor = detail::make_tensor<CustomSparseImpl>(layout, s,
      self.type_set(),
      self.dtype(),
      self.device()
  );
  auto t = torch::autograd::make_variable(tensor);
  torch::jit::push(*stack, t);
}

REGISTER_SPARSE(coo, "to_sparse", to_coo);

} // namespace at
