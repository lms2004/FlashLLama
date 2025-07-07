#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"

#include <chrono> // 添加高精度计时器

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output, double& first_token_latency, double& avg_token_latency) {
  using Clock = std::chrono::steady_clock; // 使用单调时钟
  
  // 计时相关变量
  Clock::time_point total_start_time;
  Clock::time_point gen_start_time;
  Clock::time_point first_token_start_time;
  Clock::time_point last_token_time;
  bool gen_phase_started = false;
  bool first_token_generated = false;
  std::vector<double> token_latencies; // 存储每个token的生成时延

  if (need_output) {
    total_start_time = Clock::now(); // 记录整体开始时间
  }

  auto tokens = model.encode(sentence); // string -> std::vector<int32_t>
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;

  // 每个tokenID → Embedding 向量 eg. 12 -> 12 * 768
  const auto& prompt_embedding = model.embedding(tokens);
  
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  // 循环预测 -> 直到达到最大步数
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      // 添加位置信息 -> 输入序列
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      // 首次进入生成阶段时记录开始时间
      if (need_output && !gen_phase_started) {
        gen_start_time = Clock::now();
        gen_phase_started = true;
      }
      
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
      
      // 记录生成阶段的token时延
      if (need_output) {
        const auto current_time = Clock::now();
        
        // 记录第一个生成token的开始时间
        if (!first_token_generated) {
          first_token_start_time = current_time;
          first_token_generated = true;
        }
        
        // 计算当前token的生成时延
        if (last_token_time.time_since_epoch().count() > 0) {
          const auto token_duration = std::chrono::duration_cast<std::chrono::microseconds>(
              current_time - last_token_time);
          const double token_latency_ms = token_duration.count() / 1000.0; // 转换为毫秒
          token_latencies.push_back(token_latency_ms);
        }
        
        last_token_time = current_time;
      }
    }

    pos += 1;
  }
  
  // 计算时延指标
  if (need_output && !token_latencies.empty()) {
    // 计算首字时延（从生成阶段开始到第一个token生成完成）
    const auto first_token_end_time = last_token_time;
    const auto first_token_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        first_token_end_time - first_token_start_time);
    first_token_latency = first_token_duration.count() / 1000.0; // 转换为毫秒
    
    // 计算平均时延
    double total_latency = 0.0;
    for (double latency : token_latencies) {
      total_latency += latency;
    }
    avg_token_latency = total_latency / token_latencies.size();
  }
  
  // 输出结果和解码
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
    
    // 计算并输出TPS
    const auto total_end_time = Clock::now();
    
    // 计算生成阶段的token数量 (排除prompt部分的token)
    const int32_t gen_token_count = words.size() - (prompt_len - 1);
    
    if (gen_phase_started && gen_token_count > 0) {
      // 计算生成阶段耗时
      const auto gen_duration = std::chrono::duration_cast<std::chrono::microseconds>(
          total_end_time - gen_start_time);
      const double gen_seconds = gen_duration.count() / 1000000.0;
      
      // 计算TPS
      const double tps = gen_token_count / gen_seconds;
      printf("\n\n⏱️  [Generation] Tokens: %d, Time: %.4f sec, TPS: %.2f tokens/sec\n",
             gen_token_count, gen_seconds, tps);
    }
    
    // 可选：计算整体TPS（包含prompt处理）
    const auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end_time - total_start_time);
    const double total_seconds = total_duration.count() / 1000000.0;
    printf("🏁 [Overall] Total time: %.4f sec\n", total_seconds);
  }
  
  return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  /*
    构造模型神经网络结构（核心）
    1. 读取模型配置文件
    2. 根据配置构建神经网络结构
    3. 为模型分配内存
  */
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
    checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_msg();
  }
  const std::string& sentence = "another day in the life of a software engineer, ";

  auto start = std::chrono::steady_clock::now();
  printf("✨ Generating...\n");
  fflush(stdout);
  
  // 添加时延测试变量
  double first_token_latency = 0.0;
  double avg_token_latency = 0.0;
  
  int steps = generate(model, sentence, 1000, true, first_token_latency, avg_token_latency);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\n⚡️ steps/s:%lf\n", static_cast<double>(steps) / duration);
  
  // 输出时延测试结果
  printf("\n📊 === 性能评估报告 ===\n");
  printf("🚀 首字时延 (Time to First Token): %.2f 毫秒\n", first_token_latency);
  printf("🔁 平均 Token 间时延 (Time Per Output Token): %.2f 毫秒\n", avg_token_latency);
  printf("========================\n");
  
  fflush(stdout);
  return 0;
}