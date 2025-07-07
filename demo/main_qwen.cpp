#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen2.h"
#include <chrono>
#include <algorithm> // for std::min

int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps,
                 bool need_output,
                 double& first_token_latency, double& avg_token_latency) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  words.push_back(next);

  auto start_time = std::chrono::steady_clock::now();
  bool first_token_produced = false;
  double first_token_time_ms = 0.0;

  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }

    if (!first_token_produced) {
      auto first_token_time = std::chrono::steady_clock::now();
      first_token_time_ms = std::chrono::duration<double, std::milli>(first_token_time - start_time).count();
      first_token_produced = true;
    }

    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  int output_token_count = std::max(pos, 1);
  avg_token_latency = (total_duration_ms - first_token_time_ms) / (output_token_count - 1);
  first_token_latency = first_token_time_ms;

  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
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

  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
    checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  const std::string& sentence = "hi!";

  auto start = std::chrono::steady_clock::now();
  printf("🤖 Generating...\n");
  fflush(stdout);

  double first_token_latency = 0.0;
  double avg_token_latency = 0.0;

  int steps = generate(model, sentence, 128, true, first_token_latency, avg_token_latency);

  auto end = std::chrono::steady_clock::now();
  auto duration_sec = std::chrono::duration<double>(end - start).count();

  double tps = steps / duration_sec;

  printf("\n📝 生成步骤 (steps): %d\n", steps);
  printf("⏱️ 总耗时 (duration): %.3f 秒\n", duration_sec);
  printf("⚡ 速度 (tokens/s): %.2f\n", tps);

  // 输出时延测试结果
  printf("\n📊 === 性能评估报告 ===\n");
  printf("🚀 首字时延 (Time to First Token): %.2f 毫秒\n", first_token_latency);
  printf("🔁 平均 Token 间时延 (Time Per Output Token): %.2f 毫秒\n", avg_token_latency);
  printf("⚡ 整体吞吐量 (Overall TPS): %.2f tokens/s\n", tps);
  printf("========================\n");

  fflush(stdout);
  return 0;
}
