
#ifndef KUIPER_INCLUDE_OP_EMBEDDING_H_
#define KUIPER_INCLUDE_OP_EMBEDDING_H_
#include <utility>
#include "layer.h"
namespace op {
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;

  /*
  move
    左值强制转换为右值引用,本身不执行资源移动
  */
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

 protected:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};

// 量化嵌入层，继承自 EmbeddingLayer
class QuantizedEmbeddingLayer : public EmbeddingLayer {
 public:
  explicit QuantizedEmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                                   int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

  int32_t get_scale_num() const {
    int32_t weight_size = vocab_size_ * dim_;
    return weight_size / get_group_size();
  }

 private:
  tensor::Tensor scales_;     // 缩放因子
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_EMBEDDING_H_
