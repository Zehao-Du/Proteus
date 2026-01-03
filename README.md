# Network-Aware Token Scheduling for LLM Streaming: System Design and Experimental Evaluation

## 📋 项目概述

本项目实现了一个**网络感知的 LLM Token 调度系统**，通过在 vLLM 中集成网络状态感知能力，优先调度网络条件良好的用户请求，从而提升整体系统的有效吞吐量（Effective Throughput）。系统包含三个核心组件：

1. **vLLM 调度器修改**：在 vLLM 引擎中集成 Network-Aware 调度逻辑
2. **Open WebUI 前端改造**：实时测量 RTT 并通过请求注入健康度参数
3. **实验验证框架**：通过对比实验验证 Network-Aware 调度的优势

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Open WebUI (Modified)                                   │  │
│  │  - RTT Measurement (每 2 秒)                              │  │
│  │  - Fetch Interception (自动注入 X-Client-RTT header)     │  │
│  │  - UI Display (实时显示网络状态)                          │  │
│  │  - WiFi Button (用户可控的网络优化开关)                    │  │
│  │  - Network Mode Store (全局状态管理)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │ HTTP Request
                             │ + X-Client-RTT header
                             │ + params.network_aware (WiFi按钮)
                             │ + vllm_xargs.health_factor
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Layer                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Open WebUI Backend (Modified)                           │  │
│  │  - 提取 X-Client-RTT header                              │  │
│  │  - 计算 health_factor = exp(-RTT / 500)                  │  │
│  │  - 注入到 vllm_xargs.health_factor                        │  │
│  │  - 静态 System Prompt 注入 (KV Cache 友好)                │  │
│  │  - 动态 User Prompt RTT 注入                              │  │
│  │  - 动态 Chunk Size 调整 (根据 RTT)                        │  │
│  │  - 应用层 Nagle 算法 (network_chunk_wrapper)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │ Forward Request
                             │ + health_factor in vllm_xargs
                             │ + Modified Messages (Prompt注入)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM Engine (Modified)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  EngineCore.add_request()                                 │  │
│  │  - 从 extra_args 提取 health_factor                        │  │
│  │  - 设置 request.health_factor                              │  │
│  │  - 传递给 Scheduler                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Priority Scheduler                                       │  │
│  │  - 根据 health_factor 调整请求优先级                       │  │
│  │  - 网络好用户优先调度                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │ Streaming Response
                             │ (SSE chunks)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transport Layer Optimization                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  network_chunk_wrapper()                                  │  │
│  │  - 拦截 SSE 流                                            │  │
│  │  - 积攒多个 chunk 合并成 TCP 包                            │  │
│  │  - 减少弱网环境下的 TCP 包数量                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │ Optimized TCP Packets
                             ▼
                         Client (浏览器)
```

---

## 🔧 技术实现细节

### 1. vLLM 引擎修改

**修改的文件列表**：
- `vllm/vllm/v1/engine/core.py` - 引擎核心，添加健康度提取逻辑
- `vllm/vllm/v1/core/sched/scheduler.py` - 调度器，支持基于健康度的优先级调度
- `vllm/vllm/v1/request.py` - 请求对象，添加 health_factor 字段

---

#### 1.1 Request 对象修改 (`vllm/vllm/v1/request.py`)

**修改位置**：第 140-154 行、第 179-191 行

##### 1.1.1 添加 health_factor 字段（140-143行）

在 `Request` 类的 `__init__` 方法中添加：

```python
# --- [NETWORK-AWARE SCHEDULING MODIFICATION START] ---
# Per-request health factor (0.0 - 1.0)
# 每个请求独立的健康度，用于 per-user 算力分配
self.health_factor: float = 1.0
# --- [NETWORK-AWARE SCHEDULING MODIFICATION END] ---
```

**作用**：为每个请求对象添加独立的健康度字段，默认值为 1.0（baseline 模式）。

##### 1.1.2 从 extra_args 提取 health_factor（179-191行）

在 `from_engine_core_request()` 类方法中添加：

```python
# 从 extra_args 中提取 health_factor（如果提供）
if request.sampling_params and hasattr(request.sampling_params, 'extra_args'):
    extra_args = request.sampling_params.extra_args
    if extra_args and isinstance(extra_args, dict):
        health_factor = extra_args.get("health_factor")
        if health_factor is not None:
            try:
                req.health_factor = float(health_factor)
            except (ValueError, TypeError):
                req.health_factor = 1.0  # 使用默认值
```

**作用**：在创建 Request 对象时，如果客户端通过 `vllm_xargs` 传递了 `health_factor`，则自动提取并设置。

##### 1.1.3 修改请求比较逻辑（295-310行）

修改 `__lt__` 方法，支持基于 health_factor 的优先级比较：

```python
def __lt__(self, other: "Request") -> bool:
    # 首先比较 priority（如果不同）
    if self.priority != other.priority:
        return self.priority > other.priority  # 高优先级优先
    
    # 然后比较 health_factor（高健康度优先）
    if abs(self.health_factor - other.health_factor) > 0.01:
        return self.health_factor > other.health_factor
    
    # 最后比较到达时间（早到的优先）
    return self.arrival_time < other.arrival_time
```

**作用**：在优先级队列中，相同优先级的请求会按照 health_factor 排序，网络好的用户请求优先被调度。

---

#### 1.2 EngineCore 修改 (`vllm/vllm/v1/engine/core.py`)

**修改位置**：第 394-476 行

##### 1.2.1 add_request() 方法中的健康度提取（425-474行）

在 `EngineCore.add_request()` 方法中添加了完整的 Network-Aware 调度逻辑：

```python
# --- [NETWORK-AWARE SCHEDULING] ---
health_from_request = False

# 检查 extra_args 中是否有 health_factor（最可靠的方法）
if request.sampling_params and hasattr(request.sampling_params, 'extra_args'):
    extra_args = request.sampling_params.extra_args
    if extra_args and isinstance(extra_args, dict) and "health_factor" in extra_args:
        health_from_request = True
        try:
            health_val = float(extra_args["health_factor"])
            request.health_factor = health_val
            logger.info(f"[NETWORK-AWARE] Request {request.request_id[:20]}... health={health_val:.3f} (from vllm_xargs)")
        except (ValueError, TypeError) as e:
            logger.warning(f"[NETWORK-AWARE] Failed to parse health_factor: {e}")
            health_from_request = False

if not health_from_request:
    # 请求中没有提供 health_factor，从 Hint Server 获取（fallback）
    user_id = self._extract_user_id(request.request_id)
    if user_id == 0:
        user_id = hash(request.request_id) % 10000 + 1
    
    with self.per_user_health_lock:
        if user_id in self.per_user_health:
            request.health_factor = self.per_user_health[user_id]
        else:
            # 查询 Hint Server（只在第一次查询）
            health = 1.0
            try:
                import requests as http_requests
                resp = http_requests.get(f"{self.hint_url}?user_id={user_id}", timeout=0.05)
                if resp.status_code == 200:
                    data = resp.json()
                    health = data.get("health", 1.0)
            except Exception:
                pass
            self.per_user_health[user_id] = health
            request.health_factor = health
# --- [END NETWORK-AWARE SCHEDULING] ---
```

**关键设计**：
1. **优先级机制**：优先使用请求中直接传递的 `health_factor`（通过 `vllm_xargs`）
2. **Fallback 机制**：如果没有提供，则从 Hint Server 获取（支持旧版客户端）
3. **缓存机制**：使用 `per_user_health` 字典缓存每个用户的健康度，避免重复查询
4. **线程安全**：使用 `per_user_health_lock` 保护共享数据

##### 1.2.2 用户 ID 提取方法（328-341行）

添加了 `_extract_user_id()` 辅助方法：

```python
def _extract_user_id(self, request_id: str) -> int:
    """从 request_id 中提取 user_id
    
    支持多种格式:
    - 'user{N}_xxx' -> N
    - 'chatcmpl-user{N}_xxx' -> N (vLLM 会添加 chatcmpl- 前缀)
    - 其他格式 -> 0
    """
    import re
    match = re.search(r'user(\d+)_', request_id)
    if match:
        return int(match.group(1))
    return 0
```

**作用**：从请求 ID 中提取用户 ID，用于查询 Hint Server 或使用缓存。

---

#### 1.3 Scheduler 修改 (`vllm/vllm/v1/core/sched/scheduler.py`)

**修改位置**：第 105 行、第 214-216 行、第 270-285 行、第 399 行

##### 1.3.1 添加全局 health_factor（105行）

在 `Scheduler.__init__()` 中：

```python
# Network-aware pacing factor (0.0 to 1.0)
self.health_factor = 1.0
```

**作用**：调度器级别的全局健康度因子（用于全局流控，当前版本主要使用 per-request 的 health_factor）。

##### 1.3.2 set_health_factor() 方法（214-216行）

```python
def set_health_factor(self, factor: float):
    """Update the health factor for network-aware pacing."""
    self.health_factor = max(0.01, min(1.0, factor))
```

**作用**：允许外部（如 Hint Server）动态更新全局健康度因子。

##### 1.3.3 schedule() 方法中的排序逻辑（270-285行）

在每次调度时，对 running 和 waiting 队列按 health_factor 排序：

```python
# --- [NETWORK-AWARE SCHEDULING] ---
# GPU 吞吐量不变，但优先调度高健康度的请求
# 高健康度请求更早进入 running，更早完成
# 低健康度请求等待，减少浪费

# 1. 对 running 队列按健康度排序
if self.running:
    self.running.sort(key=lambda r: -r.health_factor)

# 2. 对 waiting 队列重新排序（基于 health_factor）
#    vLLM 使用 heapq，需要重新构建堆
if hasattr(self.waiting, '_heap') and self.waiting:
    import heapq
    heapq.heapify(self.waiting._heap)
# --- [END NETWORK-AWARE SCHEDULING] ---
```

**关键机制**：
1. **Running 队列排序**：正在运行的请求按 health_factor 降序排列，高健康度请求优先处理
2. **Waiting 队列重排**：使用 `heapq.heapify()` 重新构建堆，确保高健康度请求在堆顶

##### 1.3.4 抢占逻辑中的 health_factor（399行）

在优先级抢占时，考虑 health_factor：

```python
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, -r.health_factor, r.arrival_time),
    )
```

**作用**：当需要抢占时，优先抢占低健康度的请求（在相同优先级下）。

---

#### 1.4 健康度计算

健康度计算公式：
```
health_factor = exp(-RTT / 500.0)
```

**映射关系**：
- **RTT < 100ms**：health_factor ≈ 0.82（网络极好）
- **RTT = 200ms**：health_factor ≈ 0.67（网络良好）
- **RTT = 500ms**：health_factor ≈ 0.37（网络较差）
- **RTT > 1000ms**：health_factor < 0.14（网络极差）

**设计原理**：
- 使用指数衰减函数，确保 RTT 越大，health_factor 越小
- 分母 500 是一个调优参数，控制衰减速度
- health_factor 范围 [0.0, 1.0]，值越大表示网络越好

---

#### 1.5 数据流图

```
客户端请求
    ↓
Open WebUI Backend
    ↓ (计算 health_factor = exp(-RTT/500))
    ↓ (注入到 vllm_xargs.health_factor)
    ↓
vLLM EngineCore.add_request()
    ↓ (从 extra_args 提取 health_factor)
    ↓ (设置 request.health_factor)
    ↓
Request 对象创建
    ↓ (health_factor 字段已设置)
    ↓
Scheduler.add_request()
    ↓ (加入 waiting 队列)
    ↓
Scheduler.schedule()
    ↓ (按 health_factor 排序)
    ↓ (高 health_factor 请求优先进入 running)
    ↓
GPU 执行（优先处理网络好的用户）
```

---

#### 1.6 关键设计决策

1. **Per-Request Health Factor**：每个请求独立的健康度，而不是全局统一值
2. **双重提取机制**：既支持从 `extra_args` 提取，也支持从 Hint Server 获取
3. **缓存优化**：使用字典缓存用户健康度，避免重复查询
4. **线程安全**：使用锁保护共享数据结构
5. **向后兼容**：如果没有提供 health_factor，默认使用 1.0（baseline 行为）

---

### 2. Open WebUI 前端修改

**文件位置**：`my-open-webui/src/routes/+layout.svelte`

#### 2.1 RTT 测速模块（604-638行）

```javascript
// RTT 测速逻辑
async function measureRTT() {
    const start = performance.now();
    try {
        await fetch('/api/version', {cache: "no-store"});
        const end = performance.now();
        const current = Math.round(end - start);
        window._currentRTT = current;
        
        // 更新 UI 变量
        rtt = current;
        if (rtt < 100) rttColor = 'text-green-500';      // 极好
        else if (rtt < 300) rttColor = 'text-yellow-500'; // 一般
        else rttColor = 'text-red-500';                  // 差
    } catch (e) {
        // ignore
    }
}

// 劫持 fetch，自动注入 RTT
const originalFetch = window.fetch;
window.fetch = async function(url, options) {
    if (url && url.toString().includes('/chat/completions')) {
        options = options || {};
        options.headers = options.headers || {};
        options.headers['X-Client-RTT'] = window._currentRTT.toString();
    }
    return originalFetch(url, options);
};

// 每 2 秒测量一次
measureRTT();
const rttInterval = setInterval(measureRTT, 2000);
```

**功能特点**：
- 每 2 秒自动测量一次 RTT（通过 `/api/version` 接口）
- 自动拦截所有发往 `/chat/completions` 的请求
- 在请求 header 中注入 `X-Client-RTT`
- 实时更新 UI 显示（右下角悬浮窗口）

#### 2.2 UI 显示组件（927-937行）

在屏幕右下角显示实时网络状态：

```svelte
<div class="fixed bottom-4 right-4 z-50 flex items-center gap-2 px-3 py-2 bg-gray-900/80 backdrop-blur rounded-lg border border-gray-700 shadow-lg select-none">
    <div class="text-xs font-mono text-gray-400">NETWORK RTT</div>
    <div class="text-sm font-bold font-mono {rttColor}">
        {rtt} ms
    </div>
    <!-- 动态信号格图标 -->
    <div class="flex items-end gap-0.5 h-3">
        <div class="w-1 bg-current {rtt < 500 ? rttColor : 'text-gray-600'} h-1 rounded-sm"></div>
        <div class="w-1 bg-current {rtt < 300 ? rttColor : 'text-gray-600'} h-2 rounded-sm"></div>
        <div class="w-1 bg-current {rtt < 100 ? rttColor : 'text-gray-600'} h-3 rounded-sm"></div>
    </div>
</div>
```

---

### 3. Open WebUI 后端修改

**文件位置**：`my-open-webui/backend/open_webui/main.py`

#### 3.1 RTT 处理逻辑（1529-1540行）

在 `chat_completion()` 函数中添加：

```python
# === Network-Aware Logic ===
import math
try:
    rtt = float(request.headers.get("X-Client-RTT", "100"))
    health = math.exp(-rtt / 500.0)
    health = max(0.0, min(1.0, health))
except:
    health = 1.0

if "vllm_xargs" not in form_data:
    form_data["vllm_xargs"] = {}
form_data["vllm_xargs"]["health_factor"] = health
# ===========================
```

**处理流程**：
1. 从请求 header 中提取 `X-Client-RTT`
2. 使用公式 `health = exp(-RTT / 500.0)` 计算健康度
3. 将 `health_factor` 注入到 `form_data["vllm_xargs"]` 中
4. 转发给 vLLM 时，vLLM 会自动提取并使用该参数

---

#### 3.2 网络优化增强功能（1710-1884行）

在 `process_chat()` 函数中实现了完整的网络优化逻辑，包括：

##### 3.2.1 应用层 Nagle 算法（1711-1751行）

实现了 `network_chunk_wrapper()` 函数，用于在应用层对 SSE 流进行打包：

```python
async def network_chunk_wrapper(original_iterator, chunk_size):
    """
    应用层 Nagle 算法的核心实现。
    拦截原始的 SSE 流，积攒 chunk_size 个数据包后，合并成一个 TCP 包发出。
    """
    buffer = b""
    count = 0
    min_buffer_size = max(8192, chunk_size * 500)  # 动态调整最小缓冲区
    
    try:
        async for chunk in original_iterator:
            buffer += chunk
            count += 1
            
            # 双重条件：达到包数量 OR 达到最小缓冲区大小
            if count >= chunk_size:
                yield buffer
                buffer = b""
                count = 0
            elif len(buffer) >= min_buffer_size and chunk_size > 5:
                yield buffer
                buffer = b""
                count = 0
        
        # 循环结束后，如果还有残留的，一次性发出
        if buffer:
            yield buffer
    except Exception as e:
        if buffer:
            yield buffer
        raise e
```

**设计原理**：
- **减少 TCP 包数量**：在弱网环境下，通过积攒多个 SSE chunk 减少 TCP 包数量，降低网络开销
- **动态缓冲区**：根据网络状况动态调整最小缓冲区大小，避免过度等待
- **双重触发条件**：既考虑包数量，也考虑数据大小，确保及时响应

##### 3.2.2 动态 Chunk Size 调整（1781-1790行）

根据 RTT 动态调整 chunk size：

```python
if client_rtt > 1000:
    dynamic_chunk_size = 20  # 极弱网更激进的打包
elif client_rtt > 300:
    dynamic_chunk_size = 8
else:
    dynamic_chunk_size = 1
```

**映射关系**：
- **RTT < 300ms**：chunk_size = 1（强网，无需打包）
- **RTT > 300ms**：chunk_size = 8（弱网，适度打包）
- **RTT > 1000ms**：chunk_size = 20（极弱网，激进打包）

##### 3.2.3 静态 System Prompt 注入（1796-1821行）

注入静态指令到 System Prompt，保持 KV Cache 命中：

```python
STATIC_SYS_INSTRUCTION = (
    "\n[System Instruction: You are network-aware. "
    "The user will provide their current Network RTT at the end of their message. "
    "If RTT > 300ms, answer concisely and strictly. "
    "If RTT < 100ms, answer comprehensively.]"
)

# 如果第一条是 system，追加指令
if messages[0].get("role") == "system":
    if "System Instruction: You are network-aware" not in messages[0]["content"]:
        messages[0]["content"] += STATIC_SYS_INSTRUCTION
else:
    # 如果没有 system，插入一条新的
    messages.insert(0, {
        "role": "system",
        "content": STATIC_SYS_INSTRUCTION.strip()
    })
```

**设计优势**：
- **KV Cache 友好**：静态指令不会变化，保证推理引擎的 Prefix Cache 命中
- **智能检测**：避免重复注入，保持 System Prompt 的整洁

##### 3.2.4 动态 User Prompt RTT 注入（1823-1836行）

将当前 RTT 值注入到用户消息中：

```python
if len(messages) > 0 and messages[-1].get("role") == "user":
    user_content = messages[-1]["content"]
    
    net_status = "Poor" if client_rtt > 300 else ("Excellent" if client_rtt < 100 else "Normal")
    
    rtt_injection = f"\n\n<network_context>\n  <rtt>{int(client_rtt)}ms</rtt>\n  <status>{net_status}</status>\n</network_context>"
    
    messages[-1]["content"] = user_content + rtt_injection
```

**设计优势**：
- **零缓存成本**：用户消息本身就是新的，注入动态数据不会破坏缓存
- **XML 格式**：使用结构化标签，让模型更容易理解网络上下文
- **状态描述**：提供网络状态（Excellent/Normal/Poor），帮助模型做出更好的决策

##### 3.2.5 流式响应拦截（1866-1883行）

在返回响应前，拦截并替换 `StreamingResponse` 的 `body_iterator`：

```python
if (
    enable_network_optimization
    and dynamic_chunk_size > 1
    and isinstance(final_response, StreamingResponse)
):
    original_iter = final_response.body_iterator
    final_response.body_iterator = network_chunk_wrapper(
        original_iter, dynamic_chunk_size
    )
```

**关键机制**：
- **条件拦截**：只在启用优化且 chunk_size > 1 时才拦截
- **透明替换**：直接替换 `body_iterator`，不影响其他逻辑
- **向后兼容**：如果未启用优化，响应流程保持不变

---

### 4. Open WebUI 前端增强功能

#### 4.1 网络模式状态管理

**文件位置**：`my-open-webui/src/lib/stores/network.ts`

创建全局状态存储：

```typescript
import { writable } from 'svelte/store';

// 默认关闭 (false)
export const networkMode = writable(false);
```

**作用**：提供全局的网络优化开关状态，供多个组件共享。

#### 4.2 WiFi 按钮 UI 组件

**文件位置**：`my-open-webui/src/lib/components/chat/MessageInput.svelte`

在聊天输入框工具栏中添加 WiFi 按钮：

```svelte
<script lang="ts">
    import { networkMode } from '$lib/stores/network';
    // ...
</script>

<!-- 在工具栏部分 -->
<button
    on:click={() => { $networkMode = !$networkMode; }}
    type="button"
    class="group p-[7px] flex gap-1.5 items-center text-sm rounded-full transition-colors duration-300 {$networkMode
        ? ' text-blue-500 dark:text-blue-400 bg-blue-50 hover:bg-blue-100'
        : 'bg-transparent text-gray-400 dark:text-gray-500 hover:bg-gray-50'}"
    title="Network Aware Mode (Weak Signal Optimization)"
>
    <svg class="w-5 h-5">
        <path d="M5 12.55a11 11 0 0 1 14.08 0" />
        <path d="M1.42 9a16 16 0 0 1 21.16 0" />
        <path d="M8.53 16.11a6 6 0 0 1 6.95 0" />
        <line x1="12" y1="20" x2="12.01" y2="20" />
    </svg>
</button>
```

**功能特点**：
- **可视化状态**：蓝色表示激活，灰色表示关闭
- **一键切换**：点击即可开启/关闭网络优化模式
- **实时反馈**：状态变化立即生效

#### 4.3 参数注入到请求

**文件位置**：`my-open-webui/src/lib/components/chat/Chat.svelte`

在发送消息时，将 `network_aware` 参数注入到请求中：

```svelte
<script lang="ts">
    import { networkMode } from '$lib/stores/network';
    // ...
    
    const submitMessage = async (...) => {
        // ...
        let params = { ...model.params };
        params.network_aware = $networkMode;  // 注入网络优化开关
        // ...
    }
</script>
```

**数据流**：
1. 用户点击 WiFi 按钮 → `$networkMode` 状态更新
2. 发送消息时 → `params.network_aware = $networkMode`
3. 后端接收 → 根据 `params.network_aware` 决定是否启用优化

---

## 🧪 实验设计

### 实验脚本：`timeline_experiment.py`

#### 实验目标

验证 Network-Aware 调度相比 Baseline 调度的优势：
- **Baseline 模式**：所有用户使用相同的优先级（health_factor = 1.0）
- **Network-Aware 模式**：根据用户网络 RTT 动态调整优先级（health_factor = exp(-RTT / 500)）

#### 核心假设

1. **GPU 生成速度固定**：无论调度策略如何，GPU 生成 token 的速度是恒定的，用满 GPU 的吞吐
2. **网络延迟影响有效吞吐**：chunk 到达客户端的时间 = GPU 生成时间 + 网络延迟
3. **优先调度网络好用户**：可以提升整体有效吞吐量（客户端视角）

#### 用户配置生成

使用混合高斯分布模拟真实的 4 类用户群体：

```python
network_clusters = [
    {'prob': 0.50, 'loc': 20,  'scale': 10,  'cat': 'very_good'},  # 极好网络 50%
    {'prob': 0.40, 'loc': 200, 'scale': 30,  'cat': 'good'},      # 普通网络 40%
    {'prob': 0.09, 'loc': 700, 'scale': 80,  'cat': 'bad'},       # 较差网络 9%
    {'prob': 0.01, 'loc': 2000,'scale': 400, 'cat': 'very_bad'}   # 极差网络 1%
]
```

#### 网络延迟模拟

在 `send_request()` 函数中模拟真实的网络延迟：

```python
# 单向延迟 = (RTT/2) + 0.5 * (RTT²)
rtt_sec = profile.rtt / 1000.0
one_way_delay = (rtt_sec / 2.0) + (0.5 * (rtt_sec ** 2))

# 上行延迟：请求发送前 sleep
await asyncio.sleep(one_way_delay)

# 下行延迟：在 synthetic_arrival_time 中加入
synthetic_arrival_time = observed_arrival_time + one_way_delay
```

#### 实验流程

1. **准备阶段**：
   - 生成用户配置（固定种子保证可重复性）
   - 固定请求到达顺序（seed=12345）

2. **执行阶段**：
   - 运行 Baseline 实验：所有用户 health_factor = 1.0
   - 等待 2 秒
   - 运行 Network-Aware 实验：health_factor = profile.health

3. **分析阶段**：
   - 计算累计有效 chunk 曲线（客户端视角）
   - 对比两种模式的性能差异
   - 生成统计报告和可视化图表

#### 关键指标

- **累计有效 Chunk 数**：客户端实际收到的 chunk 数量（考虑网络延迟）
- **TTFT (Time To First Token)**：从请求发送到收到第一个 chunk 的时间
- **ECPS (Effective Chunks Per Second)**：有效吞吐量
- **性能差距**：Network-Aware 相比 Baseline 的领先量

---

## 📊 实验结果

### 实验配置

- **用户数量**：8192
- **vLLM 并发度**：256（max_num_seqs）
- **客户端并发度**：2048
- **目标 QPS**：50.0（Poisson 到达模式）
- **最大 Token 数**：50

### 输出结果

实验会生成两份报告和两张图表：

1. **全部用户报告** (`timeline_comparison_all.png`)
   - 包含所有 4 类用户（very_good, good, bad, very_bad）
   - 展示整体性能提升

2. **核心用户报告** (`timeline_comparison_core.png`)
   - 仅包含网络较好的用户（very_good + good，约 90%）
   - 展示对主要用户群体的优化效果

### 图表说明

每张图表包含 4 个子图：

1. **GPU 视角**：两种模式应完全相同（GPU 生成速度固定）
2. **客户端视角**：Network-Aware 应始终高于 Baseline
3. **性能差距**：Network-Aware 的领先量（绿色为正，红色为负）
4. **有效吞吐**：ECPS 随时间的变化

---

## 🚀 部署指南

### 1. 环境准备

#### 1.1 安装 Node.js（用于编译 Open WebUI 前端）

```bash
# 安装 NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 安装 Node.js 20
nvm install 20
nvm use 20
```

#### 1.2 克隆和编译 Open WebUI

```bash
# 克隆仓库
cd ~
git clone https://github.com/open-webui/open-webui.git my-open-webui
cd my-open-webui

# 安装依赖
npm install --legacy-peer-deps

# 编译前端
npm run build
```

### 2. 修改代码

#### 2.1 修改 Open WebUI 前端

**文件 1**：`my-open-webui/src/routes/+layout.svelte`
- 添加 RTT 测速逻辑（604-638行）
- 添加 UI 显示组件（927-937行）

**文件 2**：`my-open-webui/src/lib/stores/network.ts`
- 创建网络模式状态存储

**文件 3**：`my-open-webui/src/lib/components/chat/MessageInput.svelte`
- 添加 WiFi 按钮（1646-1651行）

**文件 4**：`my-open-webui/src/lib/components/chat/Chat.svelte`
- 注入 `network_aware` 参数到请求（1937行）

#### 2.2 修改 Open WebUI 后端

编辑 `my-open-webui/backend/open_webui/main.py`：
- 在 `chat_completion()` 函数中添加 RTT 处理逻辑（1529-1540行）
- 在 `process_chat()` 函数中添加网络优化逻辑（1710-1884行）：
  - 应用层 Nagle 算法（1711-1751行）
  - 动态 Chunk Size 调整（1781-1790行）
  - 静态 System Prompt 注入（1796-1821行）
  - 动态 User Prompt RTT 注入（1823-1836行）
  - 流式响应拦截（1866-1883行）

#### 2.3 修改 vLLM 引擎

修改以下文件：
- `vllm/vllm/v1/engine/core.py`：在 `add_request()` 方法中添加健康度提取逻辑（425-474行），取消注释锁初始化（225-226行）
- `vllm/vllm/v1/core/sched/scheduler.py`：支持基于 health_factor 的优先级调度
- `vllm/vllm/v1/request.py`：添加 health_factor 字段支持

### 3. 启动服务

#### 3.1 启动 vLLM 后端

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --max-num-seqs 256 \
    --scheduling-policy priority
```

**注意**：确保使用的是修改后的 Network-Aware 版本 vLLM。

#### 3.2 启动 Open WebUI 前端

**如果容器已存在（重启服务器后）**：
```bash
docker start open-webui
```

**如果是第一次运行或容器被删除**：
```bash
docker run -d \
  -p 8080:8080 \
  -v open-webui-data:/app/backend/data \
  -v /home/argustest/my-open-webui/backend:/app/backend \
  -v /home/argustest/my-open-webui/build:/app/build \
  -e OPENAI_API_BASE_URL=http://172.17.0.1:8000/v1 \
  -e OPENAI_API_KEY=EMPTY \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

**进入容器调试**（如需要）：
```bash
docker exec -it open-webui bash
```

#### 3.3 启动内网穿透（可选，用于外网访问）

使用 Cloudflare Tunnel 实现内网穿透：

##### 3.3.1 快速启动（临时网址）

```bash
# 启动 Cloudflare Tunnel（临时网址）
nohup cloudflared tunnel --url http://127.0.0.1:8080 > tunnel.log 2>&1 &

# 查看生成的公网链接
grep "trycloudflare.com" tunnel.log
```

**说明**：
- 每次启动内网穿透，得到的网址可能不一样（临时网址）
- 生成的链接（例如 `https://happy-xx-xx.trycloudflare.com`）可以分享给任何人访问

##### 3.3.2 自定义域名部署（持久化）

**步骤 1：创建隧道**

```bash
bash cloudflare_tunnel_setup.sh
```

**步骤 2：配置域名**

1. 登录 Cloudflare Dashboard: https://dash.cloudflare.com
2. 选择域名（例如 `riverli1616.uk`）
3. 进入 'Zero Trust' > 'Networks' > 'Tunnels'
4. 找到隧道 `open-webui`，点击 'Configure'
5. 在 'Public Hostname' 中添加：
   - Subdomain: `@` (或留空)
   - Domain: `riverli1616.uk`
   - Service: `http://localhost:8080`

**步骤 3：设置系统服务（推荐）**

```bash
sudo bash setup_cloudflare_service.sh
```

**服务管理命令**：
```bash
# 查看服务状态
sudo systemctl status cloudflared-tunnel

# 查看日志
tail -f ~/cloudflare_tunnel.log

# 重启服务
sudo systemctl restart cloudflared-tunnel

# 停止服务
sudo systemctl stop cloudflared-tunnel
```

**优势**：
- ✅ 断开 SSH 后继续运行
- ✅ 服务器重启后自动启动
- ✅ 进程崩溃后自动重启
- ✅ 系统级监控和管理

#### 3.4 重新编译前端（修改代码后）

如果修改了前端代码，需要重新编译：

```bash
cd /home/argustest/my-open-webui
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20
npm run build
```

然后重启容器使修改生效：

```bash
docker restart open-webui
```

### 4. 运行实验

```bash

python timeline_experiment.py \
    --vllm-url http://localhost:8000/v1 \
    --num-users 8192 \
    --max-tokens 50 \
    --concurrency 256 \
    --client-concurrency 2048 \
    --qps 50.0
```

---

## 📈 预期效果

### 1. 系统运行效果

- **Open WebUI 界面**：
  - 右下角显示实时 RTT 和网络状态
  - 聊天输入框旁显示 WiFi 按钮（蓝色=激活，灰色=关闭）
- **vLLM 日志**：显示每个请求的 health_factor 值
- **调度行为**：网络好的用户请求优先被调度
- **传输优化**：弱网环境下，TCP 包数量显著减少（通过应用层 Nagle 算法）
- **Prompt 优化**：模型根据网络状况自动调整回复长度（通过 Prompt 注入）

### 2. 实验结果

- **累计有效 Chunk 曲线**：Network-Aware 应始终高于 Baseline
- **TTFT 改善**：Network-Aware 的平均 TTFT 应低于 Baseline
- **有效吞吐提升**：ECPS 应显著提升

---



## 📝 技术亮点

1. **端到端集成**：从客户端 RTT 测量到服务端调度决策的完整链路
2. **实时感知**：前端每 2 秒自动更新网络状态
3. **自动注入**：通过 Fetch 拦截实现零侵入的参数注入
4. **可视化展示**：实时显示网络状态，提升用户体验
5. **科学验证**：通过对比实验验证系统优势
6. **应用层优化**：实现应用层 Nagle 算法，减少 TCP 包数量
7. **智能 Prompt 工程**：静态 System Prompt + 动态 User Prompt，兼顾 KV Cache 和网络感知
8. **用户可控**：提供 WiFi 按钮，用户可手动开启/关闭网络优化
9. **多维度优化**：同时优化调度层（vLLM）和传输层（TCP chunking）
10. **生产级部署**：支持 Cloudflare Tunnel 和 systemd 服务，确保服务稳定性

---

## 🎯 未来改进方向

1. **动态调整**：根据实时网络状态动态调整健康度计算参数
2. **多维度感知**：不仅考虑 RTT，还考虑丢包率、带宽等因素
3. **自适应调度**：根据系统负载自动调整调度策略
4. **性能优化**：减少 RTT 测量开销，优化调度算法效率
5. **智能 Chunk Size**：根据历史网络状况和当前负载动态调整 chunk size
6. **A/B 测试框架**：支持对比不同优化策略的效果
7. **监控和告警**：集成 Prometheus/Grafana，实时监控网络优化效果
8. **多模型支持**：针对不同模型（GPT、Claude、Llama 等）优化 Prompt 注入策略

---

## 📚 参考文献

- vLLM: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- Open WebUI: [https://github.com/open-webui/open-webui](https://github.com/open-webui/open-webui)
- SvelteKit: [https://kit.svelte.dev/](https://kit.svelte.dev/)

---

## 👥 作者

本项目为计算机网络课程实验项目，实现了网络感知的 LLM Token 调度系统，并实机部署。

---

## 📄 License

本项目仅供学习和研究使用。

