#!/usr/bin/env python3
"""
真实 TCP 包数量测试脚本
使用 tcpdump 统计实际的 TCP 包数量（传输层）
"""

import asyncio
import httpx
import json
import time
import subprocess
import sys
import signal
import os

# ================= 🔧 配置区域 =================
BASE_URL = "http://localhost:8080" 
USER_EMAIL = "lbxhaixing154@sjtu.edu.cn"
USER_PASSWORD = "6933396li"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# 使用更长的 prompt 来生成更多内容，让效果更明显
PROMPT = "请详细解释一下量子力学的基本原理，包括波粒二象性、不确定性原理、量子纠缠等核心概念，每个概念至少用100字说明。"
# ==========================================================

async def login_and_get_token():
    """自动登录获取 Token"""
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
                print(f"✅ 登录成功! 获取到 Token: {token[:10]}...")
                return token
            else:
                print(f"❌ 登录失败: HTTP {resp.status_code}")
                return None
        except Exception as e:
            print(f"❌ 连接服务器失败: {e}")
            return None

def count_tcp_packets(port=8080, duration=60):
    """使用 tcpdump 统计 TCP 包数量（仅统计发送到客户端的包）"""
    try:
        # 检查 tcpdump 是否可用
        subprocess.run(["which", "tcpdump"], check=True, capture_output=True)
        
        # 启动 tcpdump 捕获指定端口的 TCP 包（仅出站，即服务器发送给客户端的）
        cmd = [
            "timeout", str(duration),
            "tcpdump", "-i", "any", 
            "-n",  # 不解析域名
            "-q",  # 安静模式
            f"tcp port {port} and tcp[tcpflags] & tcp-push != 0",  # 只统计有数据的包
            "-c", "10000"  # 最多捕获10000个包
        ]
        
        print(f"   📡 启动 tcpdump 监控端口 {port}...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    except FileNotFoundError:
        print("   ⚠️  tcpdump 未安装，无法统计 TCP 包数量")
        return None
    except Exception as e:
        print(f"   ⚠️  tcpdump 启动失败: {e}")
        return None

async def run_test_case(token: str, name: str, simulated_rtt: int, enable_optimization: bool):
    print(f"\n🚀 开始测试场景: [{name}]")
    print(f"   配置: RTT={simulated_rtt}ms | Optimization={'ON' if enable_optimization else 'OFF'}")

    # 启动 tcpdump
    tcpdump_process = count_tcp_packets(8080, 60)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Client-RTT": str(simulated_rtt)
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "params": {
            "network_aware": enable_optimization
        }
    }

    chunks_received = 0
    total_bytes = 0
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", f"{BASE_URL}/api/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    print(f"❌ 请求失败: HTTP {response.status_code}")
                    if tcpdump_process:
                        tcpdump_process.terminate()
                    return None

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]": 
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunks_received += 1
                                total_bytes += len(line.encode('utf-8'))
                                print(".", end="", flush=True)
                        except: 
                            pass
        except Exception as e:
            print(f"\n❌ 网络错误: {e}")
            if tcpdump_process:
                tcpdump_process.terminate()
            return None

    end_time = time.time()
    
    # 停止 tcpdump 并统计包数量
    tcp_packet_count = None
    if tcpdump_process:
        try:
            tcpdump_process.terminate()
            stdout, stderr = tcpdump_process.communicate(timeout=5)
            # 统计输出行数（每行一个包）
            tcp_packet_count = len([line for line in stdout.split('\n') if line.strip() and 'packets' not in line.lower()])
        except:
            pass
    
    if chunks_received == 0:
        print("\n⚠️ 0 数据包，测试无效")
        return None

    print("\n✅ 完成!")
    
    return {
        "name": name,
        "sse_chunks": chunks_received,  # SSE 数据包数量（应用层）
        "tcp_packets": tcp_packet_count,  # TCP 包数量（传输层）
        "total_bytes": total_bytes,
        "total_time": end_time - start_time
    }

async def main():
    # 检查权限
    if os.geteuid() != 0:
        print("⚠️  注意: 需要 root 权限运行 tcpdump")
        print("   请使用: sudo python3 test_tcp_real.py")
        print("   或者使用改进版脚本: python3 test_tcp_improved.py")
        print("")
    
    token = await login_and_get_token()
    if not token:
        sys.exit(1)

    print("========================================")
    print("   Open WebUI 网络感知模式对比测试 (TCP 层)")
    print("========================================")
    print("\n📊 说明:")
    print("   - 测试场景: 弱网 (RTT=2000ms) 下，对比开启优化 vs 关闭优化")
    print("   - SSE 包数量: 应用层的数据包（data: ... 格式）")
    print("   - TCP 包数量: 传输层的实际网络包（这才是打包器影响的）")
    print("   - 如果打包器工作，优化后 TCP 包数量应该明显减少，耗时也应该减少")
    print("")

    # 测试场景：
    # 1. 弱网 + 开启优化（我们的方案）
    # 2. 弱网 + 关闭优化（对比基准）
    result_slow_optimized = await run_test_case(token, "弱网 (RTT=2000ms) + 优化开启", 2000, True)
    if not result_slow_optimized: 
        return 

    result_slow_no_optimization = await run_test_case(token, "弱网 (RTT=2000ms) + 优化关闭", 2000, False)
    if not result_slow_no_optimization: 
        return

    print("\n\n📊 ========== 测试结果对比 ==========")
    print(f"{'指标':<30} | {'弱网+优化关闭 (基准)':<25} | {'弱网+优化开启 (我们的方案)':<25}")
    print("-" * 90)
    print(f"{'SSE 包数量 (应用层)':<30} | {result_slow_no_optimization['sse_chunks']:<25} | {result_slow_optimized['sse_chunks']:<25}")
    
    if result_slow_no_optimization['tcp_packets'] and result_slow_optimized['tcp_packets']:
        print(f"{'TCP 包数量 (传输层)':<30} | {result_slow_no_optimization['tcp_packets']:<25} | {result_slow_optimized['tcp_packets']:<25}")
        
        reduction = (1 - result_slow_optimized['tcp_packets'] / result_slow_no_optimization['tcp_packets']) * 100 if result_slow_no_optimization['tcp_packets'] > 0 else 0
        print(f"{'TCP 包减少比例':<30} | {'-':<25} | {reduction:.1f}%")
    else:
        print(f"{'TCP 包数量 (传输层)':<30} | {'需要 root 权限':<25} | {'需要 root 权限':<25}")
    
    print(f"{'总字节数':<30} | {result_slow_no_optimization['total_bytes']:<25} | {result_slow_optimized['total_bytes']:<25}")
    print(f"{'总耗时 (秒)':<30} | {result_slow_no_optimization['total_time']:<25.2f} | {result_slow_optimized['total_time']:<25.2f}")
    
    # 计算时间节省
    if result_slow_no_optimization['total_time'] > 0:
        time_saved = result_slow_no_optimization['total_time'] - result_slow_optimized['total_time']
        time_improvement = (time_saved / result_slow_no_optimization['total_time']) * 100
        print(f"{'时间节省 (秒)':<30} | {'-':<25} | {time_saved:.2f} ({time_improvement:+.1f}%)")
    
    # 分析
    print("\n🔍 ========== 分析 ==========")
    
    if result_slow_no_optimization['tcp_packets'] and result_slow_optimized['tcp_packets']:
        if result_slow_optimized['tcp_packets'] < result_slow_no_optimization['tcp_packets'] * 0.7:
            print("✅ 打包器工作正常！优化后 TCP 包数量明显减少")
            print(f"   优化前 TCP 包: {result_slow_no_optimization['tcp_packets']}")
            print(f"   优化后 TCP 包: {result_slow_optimized['tcp_packets']}")
            print(f"   减少: {result_slow_no_optimization['tcp_packets'] - result_slow_optimized['tcp_packets']} 个包 ({reduction:.1f}%)")
        else:
            print("⚠️  TCP 包数量差异不明显")
    else:
        print("⚠️  无法统计 TCP 包数量（需要 root 权限）")
        print("   建议使用 sudo 运行此脚本")
    
    # 时间分析
    if result_slow_no_optimization['total_time'] > 0:
        time_saved = result_slow_no_optimization['total_time'] - result_slow_optimized['total_time']
        if time_saved > 0.1:
            print(f"\n✅ 优化效果显著！")
            print(f"   优化前耗时: {result_slow_no_optimization['total_time']:.2f} 秒")
            print(f"   优化后耗时: {result_slow_optimized['total_time']:.2f} 秒")
            print(f"   节省时间: {time_saved:.2f} 秒 ({(time_saved / result_slow_no_optimization['total_time']) * 100:.1f}%)")
        elif time_saved > 0:
            print(f"\n✅ 优化有效果，但差异较小")
            print(f"   优化前耗时: {result_slow_no_optimization['total_time']:.2f} 秒")
            print(f"   优化后耗时: {result_slow_optimized['total_time']:.2f} 秒")
            print(f"   节省时间: {time_saved:.2f} 秒")
            print(f"   注意: 在真实弱网环境（RTT=2000ms）下，效果会更明显")
        else:
            print(f"\n⚠️  时间差异不明显")
            print(f"   可能原因:")
            print(f"   1. 本地测试环境，RTT 是模拟的，不是真实的网络延迟")
            print(f"   2. 在真实弱网环境下，减少 TCP 包数量会带来更明显的速度提升")
            print(f"   3. TCP 包减少主要影响的是网络往返时间，本地测试无法完全模拟")
    
    if abs(result_slow_no_optimization['total_bytes'] - result_slow_optimized['total_bytes']) < result_slow_no_optimization['total_bytes'] * 0.1:
        print("\n✅ 总字节数相近，说明内容相同，优化不影响内容完整性")
    else:
        print(f"\n⚠️  总字节数差异: {result_slow_no_optimization['total_bytes']} vs {result_slow_optimized['total_bytes']}")
        print("   可能是模型响应不同（RTT 注入影响了输出）")

if __name__ == "__main__":
    asyncio.run(main())

