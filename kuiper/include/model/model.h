#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_H_
#include <op/embedding.h>
#include <map>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

namespace model {
class Model {
 public:
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model = false);

  virtual base::Status init(base::DeviceType device_type) = 0;

  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;

  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual bool is_sentence_ending(int32_t token_idx) const;

  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

  // 新增：获取配置信息
  const TransformerConfig* get_config() const { return config_.get(); }
  
  // 新增：是否为量化模型
  bool is_quantized() const { return is_quant_model_; }
  
  // 新增：获取量化类型
  QuantizationType get_quantization_type() const { 
    return config_ ? config_->quant_type_ : QuantizationType::kNone; 
  }
  
  // 新增：获取文件格式版本
  FileFormatVersion get_file_format_version() const { 
    return config_ ? config_->file_version_ : FileFormatVersion::kLegacy; 
  }
  
  // 新增：获取分组大小
  int32_t get_group_size() const { return group_size_; }

 protected:
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  virtual base::Status read_model_file();

  virtual base::Status create_encode_layer();

  virtual base::Status gen_model_from_file();

  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  virtual base::Status generate_model_infos(const QuantizedModelConfig& config) const;

  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

 private:
  virtual void init_mem() = 0;

  virtual base::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
