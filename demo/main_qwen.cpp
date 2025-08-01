#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen2.h"
#include <chrono>
#include <algorithm> // for std::min

int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps,
                 bool need_output,
                 double& first_token_latency, double& avg_token_latency) {
  // 🧾 对输入句子进行编码（tokenize）
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty."; // ❗输入不能为空

  int32_t pos = 0;
  int32_t next = tokens.at(pos);  // 🧱 当前 token（初始化为第一个 token）
  bool is_prompt = true;          // 📌 是否在 prompt 阶段
  const auto& prompt_embedding = model.embedding(tokens); // 🔡 获取 prompt 的 embedding
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos); // 📍 位置张量

  std::vector<int32_t> words;
  words.push_back(next); // 🧠 收集生成的词（包括 prompt）

  auto start_time = std::chrono::steady_clock::now(); // ⏱️ 记录开始时间
  bool first_token_produced = false; // ✅ 是否已经生成第一个 token
  double first_token_time_ms = 0.0;

  // 🔁 开始生成 loop（直到生成总长度）
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos; // 📍 设置当前位置信息

    if (pos < prompt_len - 1) {
      // 🟡 仍在 prompt 区间：使用 prompt embedding
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next); // 🤖 预测下一个 token
    } else {
      // 🟢 进入生成阶段（非 prompt）
      is_prompt = false;
      tokens = std::vector<int32_t>{next}; // 🔁 上一个 token 作为当前输入
      const auto& token_embedding = model.embedding(tokens); // 🔡 获取其 embedding
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next); // 🤖 预测下一个 token
    }

    // 🕓 记录首 token latency
    if (!first_token_produced) {
      auto first_token_time = std::chrono::steady_clock::now();
      first_token_time_ms = std::chrono::duration<double, std::milli>(first_token_time - start_time).count();
      first_token_produced = true;
    }

    // 🛑 判断是否为结束标志符（如句号、<eos>等）
    if (model.is_sentence_ending(next)) {
      break;
    }

    // 📌 更新 token 输出结果
    if (is_prompt) {
      next = tokens.at(pos + 1); // prompt 阶段继续读 prompt 中的 token
      words.push_back(next);
    } else {
      words.push_back(next);     // 生成阶段记录新 token
    }

    pos += 1;
  }

  auto end_time = std::chrono::steady_clock::now(); // ⏱️ 生成结束
  double total_duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  // 📊 统计性能数据
  int output_token_count = std::max(pos, 1);
  avg_token_latency = (total_duration_ms - first_token_time_ms) / (output_token_count - 1);
  first_token_latency = first_token_time_ms;

  // 🖨️ 输出最终生成的文本
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }

  return std::min(pos, total_steps); // ✅ 返回实际生成的 token 数
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
  const std::string& sentence = R"(给我讲一个很长的故事)";


  auto start = std::chrono::steady_clock::now();
  printf("🤖 Generating...\n");
  fflush(stdout);

  double first_token_latency = 0.0;
  double avg_token_latency = 0.0;

  int steps = generate(model, sentence, 1024, true, first_token_latency, avg_token_latency);

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
