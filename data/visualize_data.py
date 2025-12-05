import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
try:
    df = pd.read_csv("net_data.csv")
except:
    print("找不到 net_data.csv，请先运行数据采集脚本。")
    exit()

# 过滤掉一些极端异常值方便绘图
# df = df[df['avg_rtt_us'] < 500000] 

plt.figure(figsize=(10, 6))

# 画散点图
# x轴: RTT, y轴: 重传
plt.scatter(df['avg_rtt_us'], df['retrans_count'], alpha=0.5, c='blue', label='Data Points')

plt.title("RTT vs Retransmission Scatter Plot")
plt.xlabel("Average RTT (us)")
plt.ylabel("Retransmission Count")
plt.grid(True, linestyle='--', alpha=0.6)

# 手动标注一下（模拟 AI 的视角）
plt.text(0, 0, "  Normal Zone", fontsize=12, color='green', fontweight='bold')
plt.text(df['avg_rtt_us'].max()*0.8, 0, "Congestion/Delay  ", fontsize=10, color='red', ha='right')
plt.text(0, df['retrans_count'].max()*0.8, "  Packet Loss/Link Error", fontsize=10, color='red')

plt.tight_layout()
plt.show()
plt.savefig("data_analysis.png")
print("图表已保存为 data_analysis.png")