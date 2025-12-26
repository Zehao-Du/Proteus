# 真实 LLM 引擎集成使用指南

本指南介绍如何使用 `real_llm_client.py` 与真实的 LLM 推理引擎（vLLM 或 Ollama）进行网络感知的 Token 生成。

## 📋 前置要求

1. **Hint Server 正在运行**
   ```bash
   cd demo
   sudo bash run_demo.sh  # 这会启动 Hint Server（在后台）
   # 或者单独启动：
   python hint_server.py --iso-model ../agent/isolation_forest.pkl \
                          --gbdt-model ../agent/gbdt_model.pkl \
                          --data-path ../data/net_data.csv
   ```

2. **eBPF 数据采集正在运行**（可选，但推荐）
   ```bash
   cd data_collection
   sudo bash collect_data.sh
   ```

3. **选择并安装 LLM 引擎**

## 🔧 选项 1：使用 Ollama（推荐，最简单）

### 安装 Ollama

```bash
# Ubuntu/Debian
curl -fsSL https://ollama.com/install.sh | sh

# 或从官网下载：https://ollama.com/download
```

### 启动 Ollama 服务

```bash
ollama serve
```

### 下载模型

```bash
# 下载一个较小的模型用于测试
ollama pull llama2

# 或下载其他模型
ollama pull mistral
ollama pull codellama
```

### 运行客户端

```bash
cd demo
python real_llm_client.py \
    --engine ollama \
    --ollama-model llama2 \
    --prompt "Tell me a short story about network optimization" \
    --max-tokens 200
```

## 🔧 选项 2：使用 vLLM（高性能）

### 安装 vLLM

```bash
pip install vllm
# 或从源码安装
# git clone https://github.com/vllm-project/vllm.git
# cd vllm && pip install -e .
```

### 启动 vLLM 服务

```bash
# 使用 OpenAI 兼容 API
python -m vllm.entrypoints.openai.api_server \
    --model <your-model-path> \
    --port 8000

# 例如使用 HuggingFace 模型
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

### 运行客户端

```bash
cd demo
python real_llm_client.py \
    --engine vllm \
    --vllm-url http://localhost:8000/v1 \
    --vllm-model default \
    --prompt "Tell me a short story about network optimization" \
    --max-tokens 200
```

## 🚀 使用启动脚本（推荐）

编辑 `run_real_llm.sh` 配置你的引擎和参数，然后运行：

```bash
cd demo
bash run_real_llm.sh
```

## 📊 观察效果

当运行 `real_llm_client.py` 时，你会看到：

1. **初始连接信息**：显示 Hint Server 连接状态和初始速率
2. **流式输出**：实时显示 LLM 生成的 Token
3. **速率信息**：每 10 个 Token 显示一次当前速率和网络健康度
4. **统计信息**：生成完成后显示总 Token 数、实际速率、目标速率和网络指标

### 示例输出

```
🔗 Connecting to Hint Server: http://localhost:5000/hint
✅ Initial rate: 45.2 tps, Health: 0.85
🚀 Using Ollama at http://localhost:11434 with model 'llama2'

📝 Prompt: Tell me a short story about network optimization
🤖 Response (rate-limited):

────────────────────────────────────────────────────────────
Once upon a time, in a digital realm where packets flowed...
[Rate: 45.2 tps, Health: 0.85] ...like rivers through...
────────────────────────────────────────────────────────────

✅ Generated 156 tokens in 3.45s
   Actual rate: 45.2 tps
   Target rate: 45.2 tps
   Network health: 0.85
   Network metrics: RTT=12000us, Retrans=0
```

## 🔍 测试网络感知效果

1. **启动数据采集和 Hint Server**
   ```bash
   # 终端 1：数据采集
   cd data_collection && sudo bash collect_data.sh
   
   # 终端 2：Hint Server
   cd demo && sudo bash run_demo.sh
   ```

2. **注入网络故障**（在另一个终端）
   ```bash
   cd data_collection
   python chaos_maker.py --delay 100ms --loss 5%
   ```

3. **运行真实 LLM 客户端**
   ```bash
   cd demo
   python real_llm_client.py --engine ollama --prompt "Your prompt here"
   ```

4. **观察速率变化**：当网络出现拥塞时，Token 生成速率会自动下降

## ⚙️ 命令行参数

```bash
python real_llm_client.py --help
```

主要参数：
- `--engine`: 选择引擎 (`vllm` 或 `ollama`)
- `--prompt`: 输入提示词
- `--hint-url`: Hint Server URL（默认：http://localhost:5000/hint）
- `--max-tokens`: 最大生成 Token 数
- `--temperature`: 采样温度
- `--disable-rate-limit`: 禁用速率限制（用于测试）

vLLM 特定参数：
- `--vllm-url`: vLLM API URL（默认：http://localhost:8000/v1）
- `--vllm-model`: 模型名称

Ollama 特定参数：
- `--ollama-url`: Ollama API URL（默认：http://localhost:11434）
- `--ollama-model`: 模型名称（默认：llama2）

## 🐛 故障排除

### Hint Server 连接失败

```
⚠️  Warning: Hint Server may not be running
```

**解决方案**：
- 确保 Hint Server 正在运行：`cd demo && sudo bash run_demo.sh`
- 或使用 `--disable-rate-limit` 禁用速率限制进行测试

### Ollama 模型未找到

```
❌ Error: Ollama request failed
```

**解决方案**：
- 确保 Ollama 服务正在运行：`ollama serve`
- 确保模型已下载：`ollama pull <model-name>`
- 检查模型名称是否正确

### vLLM 连接失败

```
❌ Error: vLLM request failed
```

**解决方案**：
- 确保 vLLM 服务正在运行
- 检查端口是否正确（默认 8000）
- 检查模型路径是否正确

## 📝 注意事项

1. **性能影响**：速率限制会在每个 Token 之间添加延迟，可能会影响生成速度
2. **网络状态**：如果 Hint Server 不可用，客户端会使用默认速率（20 tps）
3. **多流支持**：客户端支持多个并发流，每个流独立跟踪速率
4. **资源要求**：vLLM 和 Ollama 都需要足够的 GPU/CPU 和内存资源

## 🔗 相关文件

- `real_llm_client.py`: 真实 LLM 客户端主程序
- `run_real_llm.sh`: 启动脚本
- `hint_server.py`: Hint Server（提供网络状态）
- `llm_simulator.py`: LLM 模拟器（用于对比测试）

