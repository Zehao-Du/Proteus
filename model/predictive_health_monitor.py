import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt

# ================= 1. 核心类：SmartTokenPacer (带完整在线学习) =================

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_len)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class AsymmetricMSELoss(nn.Module):
    def __init__(self, penalty=15.0): 
        super().__init__()
        self.penalty = penalty

    def forward(self, pred, target):
        error = target - pred
        # 严重惩罚低估（预测值 < 真实值），因为这会导致算力过度分配引发拥塞
        loss = torch.where(error > 0, error**2 * self.penalty, error**2)
        return torch.mean(loss)

class SmartTokenPacer:
    def __init__(self, 
                 model_path=None, 
                 input_features=7, 
                 pred_len=10, 
                 learning_rate=0.001):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_len = pred_len
        self.seq_len = 10
        
        # 模型初始化
        self.model = MultiStepLSTM(input_features, 256, 2, pred_len).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 在线学习组件：切换为非对称损失
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = AsymmetricMSELoss(penalty=20.0) 
        
        # 经验池 & 延迟队列
        self.memory = deque(maxlen=2000)
        self.pending_queue = deque() # 存储 (seq, timestamp) 等待未来验证
        self.batch_size = 32
        
        # 状态追踪
        self.input_buffer = deque(maxlen=self.seq_len)
        self.min_rtt_window = deque(maxlen=200)
        
        # 平滑状态
        self.smoothed_score = 1.0
        self.smoothed_pred_rtt = None
        
        # 归一化参数 (Demo中动态更新，实际部署应固定)
        self.scaler_mean = np.zeros(input_features)
        self.scaler_scale = np.ones(input_features)

    def set_scaler(self, mean, scale):
        self.scaler_mean = np.array(mean)
        self.scaler_scale = np.array(scale)
        
    def _update_baseline(self, rtt):
        # 🔧 修复点：忽略小于 5ms (5000us) 的非法值，防止基准线被 0 污染
        if rtt > 5000:
            self.min_rtt_window.append(rtt)
        
    def get_baseline(self):
        # 🔧 修复点：强制设置最低物理基准为 30ms，适应公网环境
        if not self.min_rtt_window:
            return 30000.0
        return max(30000.0, min(self.min_rtt_window))

    def step(self, current_metrics):
        """
        Args:
            current_metrics: [log_rtt, rtt_diff, ...]
        Returns:
            health_score (0-1), pred_rtt (float)
        """
        # 1. 准备数据
        raw_feats = np.array(current_metrics)
        norm_feats = (raw_feats - self.scaler_mean) / self.scaler_scale
        
        # 解析真实 RTT (用于 Label 和 Baseline)
        current_real_rtt = np.expm1(raw_feats[0])
        self._update_baseline(current_real_rtt)
        
        self.input_buffer.append(norm_feats)
        
        # === 修复点在这里 ===
        # 如果数据不足 10 步，为了保证返回值格式一致，必须返回两个值
        if len(self.input_buffer) < self.seq_len:
            # 返回 (默认满分, 默认预测值为0)
            return 1.0, 0.0  
            
        current_seq = np.array(self.input_buffer)

        # ==========================================
        # 1. 在线学习逻辑 (Online Learning)
        # ==========================================
        
        # A. 存入待验证队列
        self.pending_queue.append(current_seq.copy())
        
        # B. 检查是否有数据"成熟"可用于训练
        if len(self.pending_queue) > self.pred_len:
            old_seq = self.pending_queue.popleft()
            target_val = norm_feats[0] 
            self.memory.append((old_seq, target_val))
            
            if len(self.memory) > self.batch_size:
                self._train()

        # ==========================================
        # 2. 推理预测 (Inference)
        # ==========================================
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
            pred_out = self.model(x_tensor).cpu().numpy()[0, -1]
            
            pred_log_rtt = pred_out * self.scaler_scale[0] + self.scaler_mean[0]
            pred_rtt = np.expm1(pred_log_rtt)

        self.model.train()

        # ==========================================
        # 3. 计算健康分 (Scoring) - 激进版
        # ==========================================
        
        # A. 预测值平滑 (降低惯性，加快响应)
        if self.smoothed_pred_rtt is None:
            self.smoothed_pred_rtt = pred_rtt
        else:
            self.smoothed_pred_rtt = 0.5 * pred_rtt + 0.5 * self.smoothed_pred_rtt
            
        # B. 动态阈值 (极限放宽版)
        base = self.get_baseline()
        # 针对当前 200ms 的环境，我们将安全区直接拉到 400ms
        threshold = max(base * 3.0, 200000.0) 
        
        diff = self.smoothed_pred_rtt - threshold
        
        # 🔧 针对 200ms 级别环境，降低敏感度，只有真正“起飞”才刹车
        val_for_sigmoid = diff / 1000.0 if abs(diff) > 1000 else diff
        
        sensitivity = 0.02  # 极低敏感度
        exponent = np.clip(sensitivity * val_for_sigmoid, -15, 15)
        raw_score = 1.0 / (1.0 + np.exp(exponent))
        
        # 🔧 极速响应恢复
        # 如果预测值正在下降，让分数回升得快一点
        if hasattr(self, 'prev_pred') and self.smoothed_pred_rtt < self.prev_pred:
            smooth_factor = 0.2
        else:
            smooth_factor = 0.5
        self.prev_pred = self.smoothed_pred_rtt
        
        self.smoothed_score = (1 - smooth_factor) * raw_score + smooth_factor * self.smoothed_score
        
        return self.smoothed_score, self.smoothed_pred_rtt

    def _train(self):
        batch = random.sample(self.memory, self.batch_size)
        bx, by = zip(*batch)
        bx = torch.FloatTensor(np.array(bx)).to(self.device)
        by = torch.FloatTensor(np.array(by)).unsqueeze(1).to(self.device) # (B, 1)
        
        self.optimizer.zero_grad()
        # 模型输出 (B, 10)，取最后一个时间步 (B, 1) 与 Label 对比
        preds = self.model(bx)[:, -1].unsqueeze(1)
        loss = self.loss_fn(preds, by)
        loss.backward()
        self.optimizer.step()


# ================= 2. 真实网络环境模拟器 (模仿 Chaos Maker) =================

class NetworkSimulator:
    """
    模拟真实的 Chaos Maker 行为：
    不是随机跳变，而是基于状态机 (State Machine) 的持续性干扰。
    """
    def __init__(self, steps):
        self.total_steps = steps
        self.current_step = 0
        
        # 状态定义
        self.STATE_NORMAL = 0
        self.STATE_CONGESTION = 1  # 带宽打满/Bufferbloat
        self.STATE_JITTER = 2      # WiFi 抖动
        
        self.current_state = self.STATE_NORMAL
        self.state_timer = 0
        
        # 物理参数
        self.base_rtt = 30 # ms
        self.queue_delay = 0 # 模拟排队积压
        
    def step(self):
        self.current_step += 1
        
        # --- 1. 状态切换逻辑 (模拟 Chaos Maker 定时切换场景) ---
        if self.state_timer <= 0:
            # 随机选择新状态，持续 50-150 步 (2.5s - 7.5s)
            rand = random.random()
            if rand < 0.5:
                self.current_state = self.STATE_NORMAL
                self.state_timer = random.randint(100, 200)
            elif rand < 0.8:
                self.current_state = self.STATE_CONGESTION
                self.state_timer = random.randint(100, 300) # 拥塞通常持续较久
            else:
                self.current_state = self.STATE_JITTER
                self.state_timer = random.randint(50, 100)
        
        self.state_timer -= 1
        
        # --- 2. 根据状态生成 RTT (物理模拟) ---
        noise = np.random.normal(0, 2)
        
        if self.current_state == self.STATE_NORMAL:
            # 正常网络：低延迟，小波动
            # 模拟队列排空
            self.queue_delay = max(0, self.queue_delay - 5) 
            rtt = self.base_rtt + noise + self.queue_delay
            
        elif self.current_state == self.STATE_CONGESTION:
            # 拥塞模式：Bufferbloat 现象
            # 队列不会瞬间变满，而是逐渐累积 (Ramp Up) -> 这才是 LSTM 能预测的关键！
            # 每次 +2ms ~ +5ms
            self.queue_delay = min(400, self.queue_delay + random.uniform(2, 5))
            rtt = self.base_rtt + self.queue_delay + noise
            
        elif self.current_state == self.STATE_JITTER:
            # 抖动模式：没有积压，但方差很大
            self.queue_delay = max(0, self.queue_delay - 5)
            jitter = random.uniform(0, 100)
            rtt = self.base_rtt + jitter + noise
            
        return max(10, rtt), self.current_state

# ================= 3. 主程序：验证 Online Learning =================

if __name__ == "__main__":
    TOTAL_STEPS = 1500
    sim = NetworkSimulator(TOTAL_STEPS)
    pacer = SmartTokenPacer(input_features=2, pred_len=10)
    pacer.set_scaler(mean=[4.0, 0.0], scale=[1.0, 1.0])
    
    history = {'real_rtt': [], 'pred_rtt': [], 'score': [], 'state': []}
    
    print("🚀 Starting Simulation...")
    prev_log_rtt = 0
    
    for t in range(TOTAL_STEPS):
        real_rtt, state = sim.step()
        log_rtt = np.log1p(real_rtt)
        rtt_diff = log_rtt - prev_log_rtt
        prev_log_rtt = log_rtt
        
        score, pred_rtt = pacer.step([log_rtt, rtt_diff])
        
        history['real_rtt'].append(real_rtt)
        history['pred_rtt'].append(pred_rtt)
        history['score'].append(score)
        history['state'].append(state)

    # ================= 可视化部分 (修改后) =================
    print("📊 Generating Dual-Axis Plot...")
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 1. 绘制背景状态带 (Background Regions)
    states = np.array(history['state'])
    # 获取 Y 轴范围以便填充整个高度
    y_max = max(max(history['real_rtt']), max(history['pred_rtt'])) * 1.1
    
    ax1.fill_between(range(TOTAL_STEPS), 0, y_max, where=(states==1), 
                     color='red', alpha=0.1, label='Congestion Zone')
    ax1.fill_between(range(TOTAL_STEPS), 0, y_max, where=(states==2), 
                     color='orange', alpha=0.1, label='Jitter Zone')

    # 2. 左轴 (Left Axis): RTT
    ax1.set_xlabel('Time Step (Simulation)', fontsize=12)
    ax1.set_ylabel('RTT (ms)', color='tab:blue', fontsize=12)
    
    # 真实 RTT (半透明，作为背景参考)
    l1, = ax1.plot(history['real_rtt'], color='tab:blue', alpha=0.3, linewidth=1, label='Real RTT')
    # 预测 RTT (深紫色虚线，展示模型追踪能力)
    l2, = ax1.plot(history['pred_rtt'], color='tab:purple', linestyle='--', linewidth=1.5, label='Pred RTT (LSTM)')
    
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, y_max)

    # 3. 右轴 (Right Axis): Health Score
    ax2 = ax1.twinx()  # 共享 X 轴
    ax2.set_ylabel('Health Score (0.0 - 1.0)', color='tab:red', fontsize=12)
    
    # 健康分 (红色粗线，醒目)
    l3, = ax2.plot(history['score'], color='tab:red', linewidth=2.5, label='Token Pacer Score')
    
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(-0.05, 1.1) # 固定范围
    
    # 辅助线 (0.5 分界线)
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    # 4. 合并图例 (Legend)
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    # 添加背景状态的图例
    import matplotlib.patches as mpatches
    patch_cong = mpatches.Patch(color='red', alpha=0.1, label='Congestion Zone')
    patch_jitt = mpatches.Patch(color='orange', alpha=0.1, label='Jitter Zone')
    
    lines.extend([patch_cong, patch_jitt])
    labels.extend(['Congestion Zone', 'Jitter Zone'])
    
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False)
    
    plt.title("Smart Token Pacer: Real-time RTT Prediction vs. Health Score", y=1.1, fontsize=14)
    plt.tight_layout()
    plt.savefig("pacer_dual_axis.png", dpi=150)
    print("✅ Plot saved to pacer_dual_axis.png")