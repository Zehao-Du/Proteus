# 多 GPU 支持说明

## 📊 当前状态

**好消息**：eBPF-TokenFlow 项目**完全支持多 GPU**！因为：

1. ✅ **vLLM 原生支持多 GPU**：vLLM 支持 Tensor Parallelism (TP)、Pipeline Parallelism (PP) 和 Data Parallelism (DP)
2. ✅ **网络感知调度兼容多 GPU**：我们的内核级调度系统（`health_factor`）在调度器层面工作，与 GPU 数量无关
3. ✅ **Hint Server 独立运行**：Hint Server 不依赖 GPU，可以独立部署

## 🚀 如何启用多 GPU

### 方法 1: 使用 Tensor Parallelism (单节点多 GPU)

**适用场景**：模型太大，单 GPU 放不下，但可以在单节点的多个 GPU 上运行。

```bash
# 使用 4 个 GPU 运行（Tensor Parallelism）
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --tensor-parallel-size 4 \
    --env VLLM_HINT_SERVER_URL=http://localhost:5000/hint
```

**参数说明**：
- `--tensor-parallel-size 4`: 使用 4 个 GPU 进行张量并行
- vLLM 会自动将模型切分到多个 GPU 上

### 方法 2: 使用 Pipeline Parallelism (多节点)

**适用场景**：模型非常大，需要跨多个节点运行。

```bash
# 8 个 GPU 总计：4 个 GPU 做 Tensor Parallel，2 个 Pipeline 阶段
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --env VLLM_HINT_SERVER_URL=http://localhost:5000/hint
```

**参数说明**：
- `--tensor-parallel-size 4`: 每个节点使用 4 个 GPU
- `--pipeline-parallel-size 2`: 使用 2 个节点（Pipeline 阶段）

### 方法 3: 使用 Data Parallelism (多副本)

**适用场景**：需要更高的并发吞吐量，运行多个模型副本。

```bash
# 使用 Ray 进行数据并行（需要先启动 Ray 集群）
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --data-parallel-size 4 \
    --env VLLM_HINT_SERVER_URL=http://localhost:5000/hint
```

## 🔧 更新启动脚本以支持多 GPU

如果你想修改启动脚本以支持多 GPU，可以添加环境变量：

```bash
# 在启动脚本中添加
export VLLM_TENSOR_PARALLEL_SIZE=4  # 使用 4 个 GPU
export VLLM_PIPELINE_PARALLEL_SIZE=1  # 单节点（不需要 Pipeline Parallel）

# 然后在启动命令中添加
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE:-1} \
    --pipeline-parallel-size ${VLLM_PIPELINE_PARALLEL_SIZE:-1} \
    --env VLLM_HINT_SERVER_URL=http://localhost:5000/hint
```

## 📈 性能影响

### 多 GPU 的优势

1. **更大的模型容量**：可以将更大的模型加载到多个 GPU 上
2. **更高的吞吐量**：Tensor Parallelism 可以加速推理
3. **更好的并发**：Data Parallelism 可以同时处理更多请求

### 网络感知调度的兼容性

✅ **完全兼容**：我们的 `health_factor` 调度机制在调度器层面工作，无论使用多少个 GPU，调度器都会根据网络健康度调整 Token 预算。

**工作原理**：
- 调度器根据 `health_factor` 限制每轮的 Token 预算
- 无论模型分布在多少个 GPU 上，调度器都会统一控制
- 多 GPU 只是加速了计算，不影响网络感知逻辑

## 🧪 测试多 GPU 配置

### 1. 检查 GPU 可用性

```bash
# 查看可用 GPU
nvidia-smi

# 或者
python3 -c "import torch; print(f'GPU 数量: {torch.cuda.device_count()}')"
```

### 2. 测试 Tensor Parallelism

```bash
# 使用 2 个 GPU 测试
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --env VLLM_HINT_SERVER_URL=http://localhost:5000/hint
```

### 3. 验证网络感知功能

```bash
# 启动 Hint Server
python3 demo/hint_server.py

# 运行客户端测试
python3 demo/real_llm_client.py \
    --engine vllm \
    --vllm-url http://localhost:8000/v1 \
    --prompt "Test multi-GPU performance"
```

观察输出中的 `Rate` 和 `Health` 指标，应该能看到网络感知的速率调整。

## ⚠️ 注意事项

### 1. GPU 内存

- 使用多 GPU 时，每个 GPU 的内存利用率会降低
- 可以通过 `--gpu-memory-utilization` 调整
- 建议：多 GPU 时可以设置更高的利用率（如 0.6-0.8）

### 2. 通信开销

- **Tensor Parallelism**：GPU 之间需要频繁通信（AllReduce），需要 NVLink 或高速互连
- **Pipeline Parallelism**：节点之间需要网络通信，需要高速网络（InfiniBand 推荐）

### 3. 模型大小

- 对于 `Qwen/Qwen3-4B-Instruct-2507`（4B 参数），通常单 GPU 就足够了
- 如果使用更大的模型（如 7B、13B、70B），才需要多 GPU

### 4. CUDA_VISIBLE_DEVICES

如果需要指定特定的 GPU：

```bash
# 只使用 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 2 \
    ...
```

## 📚 参考资源

- [vLLM 并行化文档](vllm/docs/serving/parallelism_scaling.md)
- [vLLM 配置选项](https://docs.vllm.ai/en/latest/serving/parallelism.html)
- [Megatron-LM Tensor Parallelism 论文](https://arxiv.org/pdf/1909.08053.pdf)

## 🎯 总结

| 特性 | 单 GPU | 多 GPU (TP) | 多 GPU (PP) | 多 GPU (DP) |
|------|--------|-------------|-------------|-------------|
| **模型容量** | 小 | 中 | 大 | 小（多副本） |
| **推理速度** | 慢 | 快 | 中等 | 快（并发） |
| **网络感知** | ✅ | ✅ | ✅ | ✅ |
| **适用场景** | 小模型 | 中等模型 | 大模型 | 高并发 |

**结论**：eBPF-TokenFlow 完全支持多 GPU，你只需要在启动 vLLM 时添加相应的并行参数即可！

