# 实验运行指令

## 前置条件

1. **确保 vLLM 已启动**，使用 priority 调度策略：
```bash
# 检查 vLLM 是否运行
ps aux | grep "vllm\|python.*api_server" | grep -v grep

# 如果未运行，启动 vLLM（示例）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --scheduling-policy priority \
    --max-num-seqs 256 \
    --port 8000
```

2. **确认 vLLM 使用 priority 调度**：
```bash
# 检查启动参数
ps aux | grep "scheduling-policy" | grep -v grep
```

## 运行实验

### 快速测试（小规模）
```bash
cd /home/v-boxiuli/eBPF-TokenFlow/demo

# 测试 100 个用户，每个生成 50 tokens
python timeline_experiment.py \
    --num-users 100 \
    --max-tokens 50 \
    --concurrency 64 \
    --client-concurrency 256
```

### 完整实验（推荐）
```bash
cd /home/v-boxiuli/eBPF-TokenFlow/demo

# 8192 个用户，每个生成 50 tokens
# --concurrency: vLLM 的 max_num_seqs（调度器限制）
# --client-concurrency: 客户端并发连接数（Semaphore 限制）
python timeline_experiment.py \
    --num-users 8192 \
    --max-tokens 50 \
    --concurrency 256 \
    --client-concurrency 2048
```

### 大规模实验（如果需要）
```bash
# 16384 个用户
python timeline_experiment.py \
    --num-users 16384 \
    --max-tokens 50 \
    --concurrency 256 \
    --client-concurrency 4096
```

## 实验参数说明

- `--num-users`: 用户数量（默认 8192）
- `--max-tokens`: 每个请求生成的最大 token 数（默认 50）
- `--concurrency`: vLLM 的 `max_num_seqs`（调度器限制，默认 256）
- `--client-concurrency`: 客户端并发连接数（Semaphore 限制，默认 2048）
- `--vllm-url`: vLLM API 地址（默认 http://localhost:8000/v1）

### 并发控制策略

**关键设计**：
- **`--concurrency`** (vLLM max_num_seqs): 限制 GPU 同时处理的请求数
- **`--client-concurrency`**: 限制客户端并发连接数，避免连接风暴

**为什么需要两个并发限制？**
1. **不能只限制到 256**：否则 backlog 卡在客户端，network-aware 调度器没有足够的选择空间
2. **不能无限并发**：否则会压垮系统（连接风暴、IO 瓶颈），GPU throughput 下降
3. **解决方案**：使用"足够大但有限"的客户端并发（如 2048），让 backlog 进入 vLLM 的 waiting 队列，同时避免系统过载

**推荐配置**：
- 小规模测试：`--client-concurrency 512`
- 完整实验：`--client-concurrency 2048`
- 大规模实验：`--client-concurrency 4096`（根据系统能力调整）

## 实验输出

实验会：
1. 运行 Baseline 模式（所有用户 health_factor = 1.0）
2. 运行 Network-Aware 模式（根据 RTT 计算 health_factor）
3. 生成对比图表：`timeline_comparison.png`
4. 输出统计信息：
   - 累积 token 到达曲线
   - ETPS（有效 token 吞吐量）
   - TTFT（Time To First Token）统计
   - 按用户类别（very_good/good/bad/very_bad）的统计

## 查看结果

```bash
# 查看生成的图表
ls -lh timeline_comparison.png

# 如果支持，可以直接打开
# xdg-open timeline_comparison.png  # Linux
# open timeline_comparison.png      # macOS
```

## 预期结果

Network-Aware 模式应该：
- **very_good 用户**：TTFT 显著降低（优先调度）
- **very_bad 用户**：TTFT 可能增加（延迟调度）
- **整体 ETPS**：应该提升（优先处理网络好的用户）

## 故障排查

### vLLM 未运行
```bash
# 检查端口是否被占用
netstat -tuln | grep 8000

# 查看 vLLM 日志
tail -f /tmp/vllm.log  # 如果有日志文件
```

### 实验失败
```bash
# 检查 vLLM API 是否可访问
curl http://localhost:8000/v1/models

# 检查实验脚本错误
python timeline_experiment.py --num-users 10 --max-tokens 10 2>&1 | tee experiment.log
```

### 性能问题
- 如果实验运行太慢，可以减少 `--num-users`
- 如果 GPU 内存不足，可以减少 `--concurrency`
- 如果请求超时，可以增加 `--max-tokens` 的生成时间

## 注意事项

1. **不再需要 Hint Server**：健康度直接通过 `vllm_xargs` 传递
2. **确保 vLLM 使用 priority 调度**：否则健康度不会生效
3. **并发控制**：
   - `--concurrency` 应该与 vLLM 启动时的 `--max-num-seqs` 一致
   - `--client-concurrency` 应该足够大，让 backlog 进入 vLLM 的 waiting 队列
   - 但不要太大，避免连接风暴和 IO 瓶颈
4. **实验时间**：8192 用户可能需要 2-5 分钟，取决于 GPU 性能
5. **GPU 内存**：确保有足够的 GPU 内存（建议至少 16GB）
6. **系统资源**：如果系统负载过高，可以降低 `--client-concurrency`

