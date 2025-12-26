# 快速启动指南

## 一键启动所有服务

使用 `start_services.sh` 脚本可以一键启动 Hint Server 和 vLLM。

### 基本用法

```bash
cd demo

# 启动服务（模型已固定为 Qwen/Qwen3-4B-Instruct-2507）
./start_services.sh start

# 查看服务状态
./start_services.sh status

# 停止服务
./start_services.sh stop

# 重启服务
./start_services.sh restart

# 查看日志
./start_services.sh logs        # 查看所有日志
./start_services.sh logs vllm   # 只查看 vLLM 日志
./start_services.sh logs hint   # 只查看 Hint Server 日志
```

### 环境变量配置

```bash
# 可选：GPU 内存利用率（默认 0.4）
export VLLM_GPU_MEMORY=0.4

# 可选：网络数据文件路径（默认 demo/train_data.csv）
export DATA_PATH=demo/train_data.csv
```

**注意**: vLLM 模型已固定为 `Qwen/Qwen3-4B-Instruct-2507`，无需设置 `VLLM_MODEL`。

### 完整示例

```bash
# 1. 启动服务（模型已固定）
cd demo
./start_services.sh start

# 2. 等待服务启动完成（vLLM 可能需要几分钟）

# 3. 检查服务状态
./start_services.sh status

# 4. 运行测试
python3 real_llm_client.py \
    --engine vllm \
    --vllm-url http://localhost:8000/v1 \
    --prompt "Hello, world!"

# 5. 停止服务
./start_services.sh stop
```

### 服务端口

- **Hint Server**: `http://localhost:5000`
- **vLLM API**: `http://localhost:8000`

### 日志文件

所有日志保存在 `demo/logs/` 目录：
- `hint_server.log` - Hint Server 日志
- `vllm.log` - vLLM 日志

### 故障排查

#### 端口被占用
如果提示端口被占用，可以：
1. 停止已运行的服务：`./start_services.sh stop`
2. 或者手动停止占用端口的进程

#### vLLM 启动失败
- 检查模型路径是否正确
- 检查 GPU 是否可用
- 查看日志：`./start_services.sh logs vllm`

#### Hint Server 启动失败
- 检查 Python 依赖是否安装
- 查看日志：`./start_services.sh logs hint`

### 后台运行

脚本会自动在后台运行服务，即使关闭终端也会继续运行。使用 `./start_services.sh stop` 来停止服务。

### 与 ETPS 实验结合使用

```bash
# 1. 启动服务
./start_services.sh start

# 2. 等待服务就绪
./start_services.sh status

# 3. 运行 ETPS 实验
sudo ./run_etps_experiment.sh

# 4. 停止服务
./start_services.sh stop
```

