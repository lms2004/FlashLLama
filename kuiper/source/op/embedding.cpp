#include "op/embedding.h"
#include "kernels/cpu/emb_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
  reset_weight_size(1);
  reset_input_size(3);  // 修改为3个输入：input_tokens, input_token_num, input_embeddings
  reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }

  base::Status status = check_tensor_with_dim(input_tensor, base::DeviceType::kDeviceCPU,
                                              base::DataType::kDataTypeInt32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_size, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

/*
  本质使用 emb_kernel 算子实现
*/
base::Status EmbeddingLayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::StatusCode::kSuccess;
}

// 量化嵌入层实现
QuantizedEmbeddingLayer::QuantizedEmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                                                 int32_t vocab_size)
    : EmbeddingLayer(device_type, dim, seq_len, vocab_size) {
  // 量化嵌入层使用 int8 权重
  data_type_ = base::DataType::kDataTypeInt8;
  // 设置量化层标志
  is_quant_layer_ = true;
}

base::Status QuantizedEmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }

  base::Status status = check_tensor_with_dim(input_tensor, base::DeviceType::kDeviceCPU,
                                              base::DataType::kDataTypeInt32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the quantized embedding layer.";
    return status;
  }

  // 检查量化权重
  status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                  vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the quantized embedding layer.";
    return status;
  }

  // 检查缩放因子
  if (!scales_.is_empty()) {
    status = check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_.size());
    if (!status) {
      LOG(ERROR) << "The scale tensor error in the quantized embedding layer.";
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_size, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the quantized embedding layer.";
    return status;
  }
  return base::error::Success();
}

base::Status QuantizedEmbeddingLayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  
  // 暂时使用普通嵌入内核，后续可以实现专门的量化嵌入内核
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::StatusCode::kSuccess;
}

}  // namespace op