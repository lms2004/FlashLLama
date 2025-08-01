#ifndef KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#include <ostream>
namespace model {

// 量化类型枚举
enum class QuantizationType {
  kNone = 0,      // 无量化
  kInt8 = 1,      // Int8 量化
  kInt4 = 2,      // Int4 量化
  kAWQ = 3,       // AWQ 量化
  kGPTQ = 4       // GPTQ 量化
};

// 文件格式版本枚举
enum class FileFormatVersion {
  kLegacy = 0,    // 旧版本格式
  kVersion1 = 1,  // 版本 1：标准 FP32
  kVersion2 = 2,  // 版本 2：Int8 量化
  kVersion3 = 3   // 版本 3：Legacy 量化
};

struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
  
  // 新增：量化相关配置
  QuantizationType quant_type = QuantizationType::kNone;
  FileFormatVersion file_version = FileFormatVersion::kLegacy;
  int32_t group_size = 64;  // 分组量化大小
  bool is_shared_classifier = true;  // 是否共享分类器权重
};

struct TransformerConfig {
  // 基础模型参数
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
  
  // 新增：量化配置
  QuantizationType quant_type_ = QuantizationType::kNone;
  FileFormatVersion file_version_ = FileFormatVersion::kLegacy;
  int32_t group_size_ = 64;
  bool is_shared_classifier_ = true;
  
  // 新增：模型信息
  std::string model_name_ = "";
  std::string model_family_ = "";  // "Qwen", "Llama", etc.
  float compression_ratio_ = 1.0f;  // 压缩比
  float precision_loss_ = 0.0f;     // 精度损失

  // 配置验证方法
  bool is_valid() const {
    return dim_ > 0 && hidden_dim_ > 0 && layer_num_ > 0 && 
           head_num_ > 0 && kv_head_num_ > 0 && vocab_size_ > 0 && seq_len_ > 0;
  }
  
  // 是否为量化模型
  bool is_quantized() const {
    return quant_type_ != QuantizationType::kNone;
  }
  
  // 获取量化信息字符串
  std::string get_quantization_info() const {
    if (!is_quantized()) return "FP32";
    
    switch (quant_type_) {
      case QuantizationType::kInt8: return "Int8";
      case QuantizationType::kInt4: return "Int4";
      case QuantizationType::kAWQ: return "AWQ";
      case QuantizationType::kGPTQ: return "GPTQ";
      default: return "Unknown";
    }
  }
  
  // 获取文件格式信息
  std::string get_file_format_info() const {
    switch (file_version_) {
      case FileFormatVersion::kLegacy: return "Legacy";
      case FileFormatVersion::kVersion1: return "Version1";
      case FileFormatVersion::kVersion2: return "Version2";
      case FileFormatVersion::kVersion3: return "Version3";
      default: return "Unknown";
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const TransformerConfig& obj) {
    os << "\n=== Model Configuration ===";
    os << "\nModel: " << obj.model_name_ << " (" << obj.model_family_ << ")";
    os << "\nFile Format: " << obj.get_file_format_info();
    os << "\nQuantization: " << obj.get_quantization_info();
    
    if (obj.is_quantized()) {
      os << "\nGroup Size: " << obj.group_size_;
      os << "\nCompression Ratio: " << obj.compression_ratio_ << "x";
      os << "\nPrecision Loss: " << (obj.precision_loss_ * 100) << "%";
    }
    
    os << "\n\n=== Architecture Parameters ===";
    os << "\nHidden Size (dim): " << obj.dim_;
    os << "\nIntermediate Size: " << obj.hidden_dim_;
    os << "\nNumber of Layers: " << obj.layer_num_;
    os << "\nNumber of Heads: " << obj.head_num_;
    os << "\nNumber of KV Heads: " << obj.kv_head_num_;
    os << "\nHead Size: " << obj.head_size_;
    os << "\nKV Dimension: " << obj.kv_dim_;
    os << "\nKV Multiplier: " << obj.kv_mul_;
    os << "\nVocabulary Size: " << obj.vocab_size_;
    os << "\nMax Sequence Length: " << obj.seq_len_;
    os << "\nShared Classifier: " << (obj.is_shared_classifier_ ? "Yes" : "No");
    os << "\nShared Weights: " << (obj.is_shared_weight_ ? "Yes" : "No");
    
    return os;
  }
};

}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
