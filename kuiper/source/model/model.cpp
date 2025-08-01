#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

/*
  从 buffers_ 对应类型 Tensor buffer 
*/
const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

// 新增：自动检测量化类型
QuantizationType Model::detect_quantization_type(const std::string& model_path) const {
  // 通过文件扩展名或内容检测量化类型
  if (model_path.find("int8") != std::string::npos || 
      model_path.find("q8") != std::string::npos) {
    return QuantizationType::kInt8;
  } else if (model_path.find("int4") != std::string::npos || 
             model_path.find("q4") != std::string::npos) {
    return QuantizationType::kInt4;
  } else if (model_path.find("awq") != std::string::npos) {
    return QuantizationType::kAWQ;
  } else if (model_path.find("gptq") != std::string::npos) {
    return QuantizationType::kGPTQ;
  }
  return QuantizationType::kNone;
}

// 新增：自动检测文件格式版本
FileFormatVersion Model::detect_file_format_version(const std::string& model_path) const {
  // 通过文件名或内容检测版本
  if (model_path.find("v2") != std::string::npos) {
    return FileFormatVersion::kVersion2;
  } else if (model_path.find("v1") != std::string::npos) {
    return FileFormatVersion::kVersion1;
  } else if (model_path.find("v3") != std::string::npos) {
    return FileFormatVersion::kVersion3;
  }
  return FileFormatVersion::kLegacy;
}

// 新增：设置模型信息
void Model::set_model_info(const std::string& model_path) {
  if (!config_) return;
  
  // 检测模型系列
  if (model_path.find("qwen") != std::string::npos) {
    config_->model_family_ = "Qwen";
  } else if (model_path.find("llama") != std::string::npos) {
    config_->model_family_ = "Llama";
  } else if (model_path.find("gpt") != std::string::npos) {
    config_->model_family_ = "GPT";
  } else {
    config_->model_family_ = "Unknown";
  }
  
  // 设置模型名称
  size_t last_slash = model_path.find_last_of('/');
  size_t last_dot = model_path.find_last_of('.');
  if (last_slash != std::string::npos && last_dot != std::string::npos) {
    config_->model_name_ = model_path.substr(last_slash + 1, last_dot - last_slash - 1);
  } else {
    config_->model_name_ = model_path;
  }
  
  // 设置压缩比和精度损失
  if (config_->is_quantized()) {
    switch (config_->quant_type_) {
      case QuantizationType::kInt8:
        config_->compression_ratio_ = 4.0f;
        config_->precision_loss_ = 0.01f;  // 1%
        break;
      case QuantizationType::kInt4:
        config_->compression_ratio_ = 8.0f;
        config_->precision_loss_ = 0.02f;  // 2%
        break;
      case QuantizationType::kAWQ:
        config_->compression_ratio_ = 8.0f;
        config_->precision_loss_ = 0.015f; // 1.5%
        break;
      case QuantizationType::kGPTQ:
        config_->compression_ratio_ = 8.0f;
        config_->precision_loss_ = 0.02f;  // 2%
        break;
      default:
        config_->compression_ratio_ = 1.0f;
        config_->precision_loss_ = 0.0f;
    }
  }
}

/*
  📦 自定义 .bin 权重文件读取流程

  .bin 权重结构（顺序布局）：
  """
    struct ModelConfig {
        int32_t dim;
        int32_t hidden_dim;
        int32_t layer_num;
        int32_t head_num;
        int32_t kv_head_num;
        int32_t vocab_size;
        int32_t seq_len;
    };
    int32_t group_size_;  // 👈 若为量化模型，额外读取该值
    ... 剩下部分是权重（float 或 int8）
  """
*/
base::Status Model::read_model_file() {
  using namespace base;

  // 🔍 路径为空直接报错
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }

  // 📂 尝试以只读方式打开文件（低级 API）
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  // 📖 用标准 C 库方式再打开一次用于 fread 操作（读取 config）
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  // 📐 读取模型结构配置 ModelConfig
  auto config = ModelConfig{};
  
  // 🚀 自动检测文件格式和量化类型
  QuantizationType detected_quant_type = detect_quantization_type(model_path_);
  FileFormatVersion detected_version = detect_file_format_version(model_path_);
  
  // 如果未指定量化类型，使用自动检测结果
  if (!is_quant_model_ && detected_quant_type != QuantizationType::kNone) {
    is_quant_model_ = true;
    LOG(INFO) << "Auto-detected quantization type: " << (int)detected_quant_type;
  }
  
  if (is_quant_model_) {
    // 🚀 版本 2 量化格式：256 字节文件头
    uint32_t magic;
    int32_t version;
    int32_t dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len;
    uint8_t shared_classifier;
    int32_t group_size;
    
    // 1. 读取魔数 (4 bytes)
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read magic number from model file.");
    }
    
    // 2. 读取版本 (4 bytes)
    if (fread(&version, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read version from model file.");
    }
    
    // 3. 读取 7 个模型参数 (28 bytes)
    if (fread(&dim, sizeof(int32_t), 1, file) != 1 ||
        fread(&hidden_dim, sizeof(int32_t), 1, file) != 1 ||
        fread(&layer_num, sizeof(int32_t), 1, file) != 1 ||
        fread(&head_num, sizeof(int32_t), 1, file) != 1 ||
        fread(&kv_head_num, sizeof(int32_t), 1, file) != 1 ||
        fread(&vocab_size, sizeof(int32_t), 1, file) != 1 ||
        fread(&seq_len, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read model parameters from model file.");
    }
    
    // 4. 读取共享分类器标志 (1 byte)
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
      return error::ModelParseError("Failed to read shared classifier flag from model file.");
    }
    
    // 5. 读取分组大小 (4 bytes)
    if (fread(&group_size, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError("Failed to retrieve the group size information from the model file.");
    }
    
    // 6. 设置 ModelConfig
    config.dim = dim;
    config.hidden_dim = hidden_dim;
    config.layer_num = layer_num;
    config.head_num = head_num;
    config.kv_head_num = kv_head_num;
    config.vocab_size = vocab_size;
    config.seq_len = seq_len;
    config.quant_type = QuantizationType::kInt8;
    config.file_version = FileFormatVersion::kVersion2;
    config.group_size = group_size;
    config.is_shared_classifier = shared_classifier;
    
    // 7. 设置 Model 类的成员变量
    group_size_ = group_size;
    
    // 8. 跳过剩余填充字节到 256 字节
    fseek(file, 256, SEEK_SET);
    
    
  } else {
    // 🚀 非量化格式：直接读取 ModelConfig
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
      return error::ModelParseError("Failed to retrieve the configuration information from the model file.");
    }
    config.quant_type = QuantizationType::kNone;
    config.file_version = FileFormatVersion::kLegacy;
  }

  // 
  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;  // ❌ 配置转化失败
  }

  // 🧮 准备权重数据存储结构体（根据是否量化选择）
  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();  // 使用 float32 权重
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();  // 使用 int8 权重
  }

  // 📏 获取权重文件大小（用于 mmap）
  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    return error::ModelParseError("Failed to retrieve the file size information from the model file.");
  }
  raw_model_data_->file_size = st.st_size;

  // 📋 打印模型信息（路径、大小、量化状态等）
  LOG(INFO) << "The tokenizer model path: " << token_path_;
  std::string tokenizer_type_str = tokenizer_type_ == TokenizerType::kEncodeBpe ? "Bpe" : "Spe";
  LOG(INFO) << "The tokenizer type: " << tokenizer_type_str;

  LOG(INFO) << "The model path: " << model_path_;
  LOG(INFO) << "The model file size: " << (raw_model_data_->file_size / (1 << 20)) << " MB";
  std::string quant_info = is_quant_model_ ? "quant" : "not quant";
  LOG(INFO) << "The model is " << quant_info << " model";

  if (config_) {
    // 设置模型信息
    set_model_info(model_path_);
    LOG(INFO) << "\nThe model info: " << *config_;
  }

  // 🔒 保存 fd 到 raw_model_data_，供 mmap 使用
  raw_model_data_->fd = fd;

  // 🧠 使用 mmap 将整个权重文件映射到内存空间（只读、私有映射）
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }

  /*
    🎯 计算权重数据的起始地址：
      - 量化模型：偏移量为 256 字节（完整文件头）
      - 非量化模型：偏移量为 ModelConfig 结构体大小
  */
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + 256;  // 版本 2 格式：256 字节文件头
  }

  // 🚨 最后检查指针是否为空
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }

  // ✅ 成功完成映射与数据准备
  return error::Success();
}


base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;

  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  
  // 新增：设置量化相关配置
  config_->quant_type_ = config.quant_type;
  config_->file_version_ = config.file_version;
  config_->group_size_ = config.group_size;
  config_->is_shared_classifier_ = config.is_shared_classifier;
  
  return base::error::Success();
}

// 1. Tokenization Layer
base::Status Model::create_encode_layer() {
  using namespace base;

  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#ifdef QWEN2_SUPPORT
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) { // 错误处理 -> create_encode_layer 失败
    return error::InternalError("Create the encode layer failed.");
  }

  // 分词器 vocab_size 验证
  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;

  // ⚙️ 初始化 Transformer 配置对象（用于存储模型结构信息）
  config_ = std::make_unique<TransformerConfig>();

  // 🧩 步骤 1：创建 tokenizer/embedding 编码层
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed! " << create_encode_status.get_err_msg();
    return create_encode_status; // ❌ 失败时返回错误状态
  }

  // 📂 步骤 2：读取模型文件（memory map 方式）
  // 主要读取所有的权重参数（embedding、attention、ffn、rmsnorm 等）
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Read model file " << model_path_ << " failed! " << mmap_status.get_err_msg();
    return mmap_status; // ❌ 读取失败返回
  }

  // 🏗️ 步骤 3：构建模型层（例如 attention 层、ffn 层、rmsnorm 层等）
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed! "
               << mmap_status.get_err_msg();
    return layer_create_status; // ❌ 创建失败返回
  }

  // ✅ 全部成功，返回成功状态
  return error::Success();
}


std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor>
Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const {
  // 实现 KV 缓存切片逻辑
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);

  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model