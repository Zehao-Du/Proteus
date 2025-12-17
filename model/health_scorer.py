import numpy as np

class NetworkHealthScorer:
    def __init__(self, sensitivity=0.05, loss_penalty_weight=5.0):
        """
        初始化健康评分器
        :param sensitivity: 控制 Sigmoid 曲线的陡峭程度 (k)
        :param loss_penalty_weight: 丢包对分数的惩罚权重
        """
        self.sensitivity = sensitivity
        self.loss_penalty = loss_penalty_weight
        
        # 动态维护的基准 RTT (也就是当前链路的物理极限 RTT)
        # 类似于 BBR 里的 min_rtt
        self.min_rtt_window = [] 
        self.window_size = 100 # 维护过去 100次采样 (约10秒) 的最小值

    def update_base_rtt(self, current_rtt):
        """
        在线更新 Base RTT。
        网络环境变了（比如从 WiFi 切到 5G），基准线也要变。
        """
        self.min_rtt_window.append(current_rtt)
        if len(self.min_rtt_window) > self.window_size:
            self.min_rtt_window.pop(0)
            
    def get_base_rtt(self):
        if not self.min_rtt_window:
            return 20.0 # 默认初始值
        return min(self.min_rtt_window)

    def calculate_score(self, predicted_rtt, predicted_loss_rate=0.0):
        """
        核心函数：将 RTT 映射为 0-1 分数
        """
        base_rtt = self.get_base_rtt()
        
        # 1. 定义容忍阈值 (Tolerance Threshold)
        # 我们允许 RTT 波动到 Base RTT 的 2倍以内都算相对健康
        # 超过 3倍 Base RTT 或超过绝对值 500ms 开始剧烈扣分
        threshold = max(base_rtt * 2.5, base_rtt + 100) 
        
        # 2. 计算延迟得分 (Sigmoid Mapping)
        # x 是 "超出的延迟"
        x = predicted_rtt - threshold
        
        # Sigmoid 变体：
        # 当 pred_rtt < threshold 时，x 为负，score 趋近 1.0
        # 当 pred_rtt > threshold 时，x 为正，score 迅速掉到 0.0
        latency_score = 1.0 / (1.0 + np.exp(self.sensitivity * x))
        
        # 3. 计算丢包惩罚
        # 丢包是严重的健康问题，直接乘法惩罚
        # 如果 predicted_loss_rate 是 0.1 (10%丢包)，分数直接打折
        loss_score = max(0.0, 1.0 - (predicted_loss_rate * self.loss_penalty))
        
        # 4. 最终得分
        final_score = latency_score * loss_score
        
        return float(final_score), base_rtt, threshold

# ================= 测试代码 =================
if __name__ == "__main__":
    scorer = NetworkHealthScorer(sensitivity=0.05)
    
    # 模拟：先跑一段正常的 30ms 网络
    print(f"{'Pred RTT':<10} | {'Base RTT':<10} | {'Score':<10} | {'Status'}")
    print("-" * 50)
    
    # 1. 正常阶段
    for rtt in [30, 32, 29, 31, 35]:
        scorer.update_base_rtt(rtt)
        score, base, _ = scorer.calculate_score(rtt)
        print(f"{rtt:<10} | {base:<10} | {score:.4f}     | ✅ Healthy")

    # 2. 拥塞发生 (Bufferbloat)，RTT 预测值开始飙升
    print("-" * 50)
    print(">>> Congestion Detected (Prediction)")
    predictions = [50, 80, 120, 200, 350, 500]
    for pred_rtt in predictions:
        # 注意：这里我们不 update base_rtt，因为这些是拥塞导致的虚高，不是物理链路变了
        score, base, thresh = scorer.calculate_score(pred_rtt)
        status = "⚠️ Degraded" if score < 0.8 else "✅ Healthy"
        if score < 0.4: status = "❌ Critical"
        
        print(f"{pred_rtt:<10} | {base:<10} | {score:.4f}     | {status}")