#!/usr/bin/env python3
"""
验证打包器是否真正工作的脚本
通过分析实际接收到的数据包大小来验证
"""

import asyncio
import httpx
import json
import time
import sys

BASE_URL = "http://localhost:8080" 
USER_EMAIL = "lbxhaixing154@sjtu.edu.cn"
USER_PASSWORD = "6933396li"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT = "请从1数到50，数字之间用逗号隔开，不要换行。"

async def login_and_get_token():
    print(f"🔑 正在尝试使用账号 {USER_EMAIL} 登录...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/auths/signin",
                json={"email": USER_EMAIL, "password": USER_PASSWORD}
            )
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("token")
                print(f"✅ 登录成功!")
                return token
            return None
        except Exception as e:
            print(f"❌ 连接服务器失败: {e}")
            return None

async def test_with_detailed_analysis(token: str, name: str, rtt: int):
    print(f"\n🚀 测试: {name} (RTT={rtt}ms)")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Client-RTT": str(rtt)
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "params": {
            "network_aware": True
        }
    }

    chunks = []
    packet_sizes = []  # 每个实际接收到的数据包大小
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", f"{BASE_URL}/api/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    print(f"❌ 请求失败: HTTP {response.status_code}")
                    return None

                # 使用 aiter_bytes 来获取原始字节流
                async for chunk_bytes in response.aiter_bytes():
                    if chunk_bytes:
                        packet_sizes.append(len(chunk_bytes))
                        # 解析 SSE 数据
                        try:
                            text = chunk_bytes.decode('utf-8')
                            for line in text.split('\n'):
                                if line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        delta = data_json.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            chunks.append(content)
                                    except:
                                        pass
                        except:
                            pass
                        print(".", end="", flush=True)
        except Exception as e:
            print(f"\n❌ 网络错误: {e}")
            return None

    end_time = time.time()
    
    if not chunks:
        print("\n⚠️ 0 数据包")
        return None

    print("\n✅ 完成!")
    
    # 统计信息
    total_packets = len(packet_sizes)
    avg_packet_size = sum(packet_sizes) / len(packet_sizes) if packet_sizes else 0
    max_packet_size = max(packet_sizes) if packet_sizes else 0
    min_packet_size = min(packet_sizes) if packet_sizes else 0
    total_bytes = sum(packet_sizes)
    
    return {
        "name": name,
        "rtt": rtt,
        "sse_chunks": len(chunks),
        "tcp_packets": total_packets,  # 实际接收到的 TCP 包数量
        "avg_packet_size": avg_packet_size,
        "max_packet_size": max_packet_size,
        "min_packet_size": min_packet_size,
        "total_bytes": total_bytes,
        "packet_sizes": packet_sizes[:20],  # 前20个包的大小
        "total_time": end_time - start_time
    }

async def main():
    token = await login_and_get_token()
    if not token:
        sys.exit(1)

    print("=" * 60)
    print("   打包器验证测试 - 分析 TCP 层数据包")
    print("=" * 60)
    print("\n📊 说明:")
    print("   - 使用 aiter_bytes() 获取原始字节流")
    print("   - 统计实际接收到的 TCP 数据包数量和大小")
    print("   - 如果打包器工作，弱网时包数量应该减少，包大小应该增加")
    print("")

    result_fast = await test_with_detailed_analysis(token, "强网", 10)
    if not result_fast:
        return

    result_slow = await test_with_detailed_analysis(token, "弱网", 2000)
    if not result_slow:
        return

    print("\n\n" + "=" * 60)
    print("   测试结果对比")
    print("=" * 60)
    print(f"{'指标':<30} | {'强网 (RTT 10)':<20} | {'弱网 (RTT 2000)':<20}")
    print("-" * 80)
    print(f"{'SSE 数据包数量':<30} | {result_fast['sse_chunks']:<20} | {result_slow['sse_chunks']:<20}")
    print(f"{'TCP 包数量 (接收到的)':<30} | {result_fast['tcp_packets']:<20} | {result_slow['tcp_packets']:<20}")
    print(f"{'平均包大小 (字节)':<30} | {result_fast['avg_packet_size']:<20.2f} | {result_slow['avg_packet_size']:<20.2f}")
    print(f"{'最大包大小 (字节)':<30} | {result_fast['max_packet_size']:<20} | {result_slow['max_packet_size']:<20}")
    print(f"{'最小包大小 (字节)':<30} | {result_fast['min_packet_size']:<20} | {result_slow['min_packet_size']:<20}")
    print(f"{'总字节数':<30} | {result_fast['total_bytes']:<20} | {result_slow['total_bytes']:<20}")
    print(f"{'总耗时 (秒)':<30} | {result_fast['total_time']:<20.2f} | {result_slow['total_time']:<20.2f}")
    
    print("\n" + "=" * 60)
    print("   分析")
    print("=" * 60)
    
    # 检查 TCP 包数量
    if result_slow['tcp_packets'] < result_fast['tcp_packets'] * 0.8:
        reduction = (1 - result_slow['tcp_packets'] / result_fast['tcp_packets']) * 100
        print(f"✅ 打包器工作正常！")
        print(f"   - TCP 包数量减少: {result_fast['tcp_packets']} → {result_slow['tcp_packets']} ({reduction:.1f}%)")
        print(f"   - 平均包大小增加: {result_fast['avg_packet_size']:.2f} → {result_slow['avg_packet_size']:.2f} 字节")
        print(f"   - 最大包大小: {result_fast['max_packet_size']} → {result_slow['max_packet_size']} 字节")
    elif result_slow['avg_packet_size'] > result_fast['avg_packet_size'] * 1.5:
        print(f"✅ 打包器可能在工作！")
        print(f"   - 平均包大小明显增加: {result_fast['avg_packet_size']:.2f} → {result_slow['avg_packet_size']:.2f} 字节")
        print(f"   - 包数量: {result_fast['tcp_packets']} → {result_slow['tcp_packets']}")
    else:
        print(f"⚠️  打包器效果不明显")
        print(f"   - TCP 包数量: {result_fast['tcp_packets']} → {result_slow['tcp_packets']}")
        print(f"   - 平均包大小: {result_fast['avg_packet_size']:.2f} → {result_slow['avg_packet_size']:.2f} 字节")
        print(f"\n   可能原因:")
        print(f"   1. httpx 的 aiter_bytes() 可能已经做了缓冲")
        print(f"   2. 需要使用 tcpdump 在更底层验证")
        print(f"   3. 网络栈的 Nagle 算法也在工作")
    
    # 显示前几个包的大小分布
    print(f"\n📦 前10个包的大小分布:")
    print(f"   强网: {result_fast['packet_sizes'][:10]}")
    print(f"   弱网: {result_slow['packet_sizes'][:10]}")

if __name__ == "__main__":
    asyncio.run(main())


