#ifndef KUIPER_INCLUDE_MODEL_CONFIG_ADAPTER_H_
#define KUIPER_INCLUDE_MODEL_CONFIG_ADAPTER_H_

#include "config.h"
#include <string>
#include <memory>

namespace model {

// 配置适配器基类
class ConfigAdapter {
public:
    virtual ~ConfigAdapter() = default;
    
    // 验证配置是否有效
    virtual bool validate_config(const TransformerConfig& config) const = 0;
    
    // 获取配置建议
    virtual std::string get_config_suggestions(const TransformerConfig& config) const = 0;
    
    // 修复配置问题
    virtual bool fix_config(TransformerConfig& config) const = 0;
    
    // 获取配置摘要
    virtual std::string get_config_summary(const TransformerConfig& config) const = 0;
};

// Int8 量化配置适配器
class Int8ConfigAdapter : public ConfigAdapter {
public:
    bool validate_config(const TransformerConfig& config) const override {
        if (!config.is_valid()) return false;
        
        // Int8 特定验证
        if (config.quant_type_ != QuantizationType::kInt8) return false;
        if (config.group_size_ <= 0 || config.group_size_ > 256) return false;
        if (config.compression_ratio_ < 3.0f || config.compression_ratio_ > 5.0f) return false;
        
        return true;
    }
    
    std::string get_config_suggestions(const TransformerConfig& config) const override {
        std::string suggestions = "Int8 Quantization Suggestions:\n";
        
        if (config.group_size_ < 32) {
            suggestions += "- Consider increasing group_size to 64 for better precision\n";
        }
        if (config.group_size_ > 128) {
            suggestions += "- Consider decreasing group_size to 64 for better compression\n";
        }
        if (config.precision_loss_ > 0.02f) {
            suggestions += "- High precision loss detected, consider recalibrating\n";
        }
        
        return suggestions;
    }
    
    bool fix_config(TransformerConfig& config) const override {
        bool fixed = false;
        
        // 修复分组大小
        if (config.group_size_ <= 0 || config.group_size_ > 256) {
            config.group_size_ = 64;
            fixed = true;
        }
        
        // 修复压缩比
        if (config.compression_ratio_ < 3.0f || config.compression_ratio_ > 5.0f) {
            config.compression_ratio_ = 4.0f;
            fixed = true;
        }
        
        return fixed;
    }
    
    std::string get_config_summary(const TransformerConfig& config) const override {
        return "Int8 Quantized Model - " + config.model_name_ + 
               " (Compression: " + std::to_string(config.compression_ratio_) + "x, " +
               "Precision Loss: " + std::to_string(config.precision_loss_ * 100) + "%)";
    }
};

// FP32 配置适配器
class FP32ConfigAdapter : public ConfigAdapter {
public:
    bool validate_config(const TransformerConfig& config) const override {
        if (!config.is_valid()) return false;
        
        // FP32 特定验证
        if (config.quant_type_ != QuantizationType::kNone) return false;
        if (config.compression_ratio_ != 1.0f) return false;
        if (config.precision_loss_ != 0.0f) return false;
        
        return true;
    }
    
    std::string get_config_suggestions(const TransformerConfig& config) const override {
        std::string suggestions = "FP32 Model Suggestions:\n";
        
        if (config.dim_ > 4096) {
            suggestions += "- Large model detected, consider quantization for memory efficiency\n";
        }
        if (config.layer_num_ > 32) {
            suggestions += "- Deep model detected, consider optimization techniques\n";
        }
        
        return suggestions;
    }
    
    bool fix_config(TransformerConfig& config) const override {
        bool fixed = false;
        
        // 确保 FP32 配置正确
        if (config.quant_type_ != QuantizationType::kNone) {
            config.quant_type_ = QuantizationType::kNone;
            fixed = true;
        }
        if (config.compression_ratio_ != 1.0f) {
            config.compression_ratio_ = 1.0f;
            fixed = true;
        }
        if (config.precision_loss_ != 0.0f) {
            config.precision_loss_ = 0.0f;
            fixed = true;
        }
        
        return fixed;
    }
    
    std::string get_config_summary(const TransformerConfig& config) const override {
        return "FP32 Model - " + config.model_name_ + 
               " (Full Precision, No Compression)";
    }
};

// 配置适配器工厂
class ConfigAdapterFactory {
public:
    static std::unique_ptr<ConfigAdapter> create_adapter(QuantizationType quant_type) {
        switch (quant_type) {
            case QuantizationType::kInt8:
                return std::make_unique<Int8ConfigAdapter>();
            case QuantizationType::kNone:
                return std::make_unique<FP32ConfigAdapter>();
            default:
                // 对于其他量化类型，返回 Int8 适配器作为默认
                return std::make_unique<Int8ConfigAdapter>();
        }
    }
    
    static std::unique_ptr<ConfigAdapter> create_adapter(const TransformerConfig& config) {
        return create_adapter(config.quant_type_);
    }
};

// 配置管理器
class ConfigManager {
public:
    ConfigManager(const TransformerConfig& config) 
        : config_(config), adapter_(ConfigAdapterFactory::create_adapter(config)) {}
    
    // 验证配置
    bool validate() const {
        return adapter_->validate_config(config_);
    }
    
    // 获取建议
    std::string get_suggestions() const {
        return adapter_->get_config_suggestions(config_);
    }
    
    // 修复配置
    bool fix_config() {
        return adapter_->fix_config(config_);
    }
    
    // 获取摘要
    std::string get_summary() const {
        return adapter_->get_config_summary(config_);
    }
    
    // 获取配置
    const TransformerConfig& get_config() const {
        return config_;
    }
    
    // 检查是否为量化模型
    bool is_quantized() const {
        return config_.is_quantized();
    }
    
    // 获取量化信息
    std::string get_quantization_info() const {
        return config_.get_quantization_info();
    }

private:
    TransformerConfig config_;
    std::unique_ptr<ConfigAdapter> adapter_;
};

} // namespace model

#endif // KUIPER_INCLUDE_MODEL_CONFIG_ADAPTER_H_ 