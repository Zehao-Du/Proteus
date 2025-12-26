# 🏭 工业级场景改进需求 - Member A 任务清单

本文档详细列出了将 TokenFlow 系统应用于工业级生产环境时，Member A（系统与内核专家）需要完成的关键任务。

## 📊 当前实现 vs 工业级需求对比

| 维度 | 当前实现 | 工业级需求 | 优先级 |
|------|---------|-----------|--------|
| **架构** | 单节点 | 分布式、高可用 | 🔴 P0 |
| **遥测** | TCP RTT/重传 | 全栈遥测（网络+硬件+应用） | 🔴 P0 |
| **容错** | 基础异常处理 | 自动恢复、降级策略 | 🔴 P0 |
| **性能** | 单流控制 | 多流并发、连接池 | 🟡 P1 |
| **监控** | 基础日志 | 指标采集、告警、追踪 | 🟡 P1 |
| **安全** | 无认证 | 认证、授权、加密 | 🟡 P1 |
| **扩展性** | 固定配置 | 动态配置、多租户 | 🟢 P2 |

---

## 🎯 Member A 核心任务（按优先级）

### 🔴 P0 - 关键任务（必须实现）

#### 1. 高级 eBPF 遥测系统

**目标**：从单一网络指标扩展到全栈遥测，为决策提供更丰富的上下文。

##### 1.1 GPU 监控集成
```python
# 需要实现：gpu_monitor.py
- 使用 NVML (NVIDIA Management Library) 收集：
  * GPU 利用率 (%)
  * GPU 内存使用 (MB/GB)
  * GPU 温度 (°C)
  * 功耗 (W)
  * 计算单元占用率
- 集成到 ebpf_collector.py，与网络指标一起输出
- 更新数据格式：net_data.csv 增加 gpu_util, gpu_mem, gpu_temp 列
```

**技术要点**：
- 安装 `pynvml` 库
- 处理多 GPU 场景（选择监控哪个 GPU）
- 处理无 GPU 环境（优雅降级）
- 采样频率：1-5 秒（避免过高开销）

**验收标准**：
- [ ] 能够实时采集 GPU 指标
- [ ] 数据写入 CSV 并与网络指标时间戳对齐
- [ ] Dashboard 能够显示 GPU 指标

##### 1.2 Socket Backlog 监控
```c
// eBPF 程序需要 Hook：
- tcp_v4_syn_recv_sock() 或 tcp_v6_syn_recv_sock()
- 读取 sk->sk_ack_backlog 和 sk->sk_max_ack_backlog
- 计算 backlog 使用率 = ack_backlog / max_ack_backlog
```

**技术要点**：
- 检测接收端拥塞（backlog 接近 max 时）
- 区分应用层处理慢 vs 网络层拥塞
- 输出到 CSV：`socket_backlog_ratio`

**验收标准**：
- [ ] eBPF 程序能够读取 socket backlog
- [ ] 数据采集正常，无性能影响
- [ ] 能够识别接收端拥塞场景

##### 1.3 HTTP/gRPC 应用层延迟追踪（可选但推荐）
```python
# 使用 eBPF uprobes 追踪：
- HTTP 请求/响应时间
- gRPC 调用延迟
- 应用层错误率
```

**技术要点**：
- 使用 `uprobe` 追踪用户空间函数
- 识别慢请求（P95/P99 延迟）
- 关联网络指标和应用指标

---

#### 2. 分布式架构与高可用

**目标**：从单节点扩展到多节点，支持水平扩展和故障恢复。

##### 2.1 Hint Server 集群化
```python
# 需要实现：
- 多实例 Hint Server（负载均衡）
- 共享状态（使用 Redis）
- 健康检查和自动故障转移
- 配置中心（Consul/etcd）
```

**架构设计**：
```
                    [Load Balancer]
                          |
        +-----------------+-----------------+
        |                 |                 |
   [Hint Server 1]  [Hint Server 2]  [Hint Server 3]
        |                 |                 |
        +-----------------+-----------------+
                          |
                    [Redis Cluster]
                          |
                    [Data Source]
```

**技术要点**：
- 使用 Redis 存储最新网络指标（替代 CSV）
- 实现 Leader Election（避免重复计算）
- 实现请求路由和负载均衡（Nginx/HAProxy）
- 实现优雅关闭和重启

**验收标准**：
- [ ] 支持 3+ 个 Hint Server 实例
- [ ] 单个实例故障不影响服务
- [ ] 负载均衡正常工作
- [ ] 数据一致性保证

##### 2.2 数据采集分布式部署
```python
# 需要实现：
- 多节点 eBPF 采集器
- 数据聚合服务（Aggregator）
- 时间同步（NTP/PTP）
- 数据去重和合并
```

**技术要点**：
- 每个节点独立运行 `ebpf_collector.py`
- Aggregator 服务收集各节点数据并合并
- 处理时钟偏差和数据延迟
- 实现数据压缩和批处理

---

#### 3. 容错与自动恢复

**目标**：系统能够自动处理故障并恢复，减少人工干预。

##### 3.1 客户端容错机制
```python
# real_llm_client.py 需要增强：
- Hint Server 连接失败：使用本地缓存速率
- 速率查询超时：使用指数退避重试
- LLM 引擎连接失败：自动重连
- 网络中断：暂停生成，等待恢复
```

**实现要点**：
- 实现本地速率缓存（最近 N 次查询的平均值）
- 实现断路器模式（Circuit Breaker）
- 实现健康检查端点
- 实现自动降级策略

##### 3.2 eBPF 采集器自动恢复
```python
# ebpf_collector.py 需要增强：
- eBPF 程序加载失败：自动重试
- 内核版本不兼容：优雅降级
- 数据写入失败：本地缓冲+重试
- 进程崩溃：systemd 自动重启
```

---

### 🟡 P1 - 重要任务（强烈推荐）

#### 4. 性能优化

##### 4.1 多流并发优化
```python
# real_llm_client.py 当前问题：
- 每个 Token 都 sleep，影响吞吐量
- 单线程处理，无法充分利用多核

# 需要改进：
- 实现 Token 批处理（batch pacing）
- 使用异步 I/O（asyncio/aiohttp）
- 实现连接池（复用 HTTP 连接）
- 实现背压控制（backpressure）
```

##### 4.2 速率控制算法优化
```python
# 当前实现：简单的 sleep 控制
# 工业级需求：
- Token Bucket 算法（更平滑的速率控制）
- 自适应窗口调整
- 预测性速率调整（基于历史趋势）
```

##### 4.3 数据采集性能优化
```python
# ebpf_collector.py 优化：
- 减少 CSV 写入频率（批量写入）
- 使用内存映射文件（mmap）
- 实现数据压缩
- 异步 I/O 处理
```

---

#### 5. 监控与可观测性

##### 5.1 指标采集系统
```python
# 需要实现：metrics_collector.py
- Prometheus 格式指标导出
- 关键指标：
  * hint_server_request_count
  * hint_server_latency_ms
  * llm_client_token_rate
  * network_rtt_us
  * network_retrans_count
  * gpu_utilization_percent
  * socket_backlog_ratio
```

##### 5.2 分布式追踪
```python
# 使用 OpenTelemetry 或 Jaeger
- 追踪请求从客户端到 LLM 引擎的完整路径
- 关联网络指标和应用指标
- 识别性能瓶颈
```

##### 5.3 日志聚合
```python
# 结构化日志（JSON 格式）
- 使用 ELK Stack 或 Loki
- 日志级别：DEBUG/INFO/WARN/ERROR
- 关键事件：速率变化、故障恢复、异常检测
```

##### 5.4 告警系统
```python
# 告警规则：
- 网络健康度 < 0.3 持续 30 秒 → 告警
- Hint Server 不可用 → 告警
- GPU 利用率 > 90% → 告警
- Socket Backlog > 80% → 告警
```

---

#### 6. 安全与认证

##### 6.1 API 认证
```python
# Hint Server 需要添加：
- API Key 认证
- OAuth2/JWT Token 认证
- Rate Limiting（防止滥用）
- IP 白名单（可选）
```

##### 6.2 数据加密
```python
# 传输加密：
- HTTPS/TLS（Hint Server）
- 敏感数据加密存储

# 数据隐私：
- 不记录用户提示词内容
- 匿名化网络指标
```

---

### 🟢 P2 - 增强功能（可选）

#### 7. 多租户支持

##### 7.1 租户隔离
```python
# 需要实现：
- 每个租户独立的速率限制
- 租户级别的资源配额
- 租户级别的监控和告警
```

#### 8. 动态配置

##### 8.1 配置热更新
```python
# 无需重启即可更新：
- 速率限制参数
- 模型路径
- 告警阈值
- 采样频率
```

---

## 📋 实施计划建议

### Phase 1: 基础增强（2-3 周）
1. GPU 监控集成
2. Socket Backlog 监控
3. 客户端容错机制
4. 基础监控指标

### Phase 2: 分布式架构（3-4 周）
1. Redis 集成
2. Hint Server 集群化
3. 负载均衡
4. 数据聚合服务

### Phase 3: 生产级特性（2-3 周）
1. 完整监控和告警
2. 安全认证
3. 性能优化
4. 文档和运维手册

---

## 🔧 技术栈建议

### 数据存储
- **Redis**: 实时指标存储（替代 CSV）
- **TimescaleDB/InfluxDB**: 历史数据存储
- **对象存储 (S3)**: 模型文件存储

### 服务发现与配置
- **Consul/etcd**: 服务注册与配置中心
- **Kubernetes**: 容器编排（如果使用 K8s）

### 监控与追踪
- **Prometheus**: 指标采集
- **Grafana**: 可视化
- **Jaeger/Zipkin**: 分布式追踪
- **ELK Stack**: 日志聚合

### 部署
- **Docker**: 容器化
- **Kubernetes**: 编排（可选）
- **Helm**: 包管理（如果使用 K8s）

---

## 📝 验收标准总结

### 功能完整性
- [ ] GPU 监控正常工作
- [ ] Socket Backlog 监控正常工作
- [ ] 分布式架构支持 3+ 节点
- [ ] 容错机制验证通过

### 性能指标
- [ ] Hint Server P99 延迟 < 50ms
- [ ] 支持 1000+ 并发客户端
- [ ] 数据采集开销 < 5% CPU

### 可靠性
- [ ] 单节点故障不影响服务
- [ ] 自动恢复时间 < 30 秒
- [ ] 99.9% 可用性（月度）

### 可观测性
- [ ] 所有关键指标可监控
- [ ] 告警系统正常工作
- [ ] 日志可查询和追踪

---

## 🎓 学习资源

- **eBPF 高级编程**: 《Linux Observability with BPF》
- **分布式系统**: 《Designing Data-Intensive Applications》
- **监控实践**: Prometheus 官方文档
- **GPU 监控**: NVIDIA NVML 文档

---

## 📞 协作接口

### 与 Member B 的接口
- **数据格式**: 定义扩展后的 CSV/JSON 格式（包含 GPU 指标）
- **模型输入**: 确保新指标能够输入到模型中

### 与 Member C 的接口
- **API 规范**: Hint Server API 版本管理
- **Dashboard**: 新指标的可视化需求
- **部署**: Docker 镜像和配置需求

---

**最后更新**: 2024
**负责人**: Member A (System & Kernel Expert)

