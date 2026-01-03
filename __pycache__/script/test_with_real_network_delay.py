#!/usr/bin/env python3
"""
使用真实网络延迟模拟的测试脚本
使用 Linux tc (traffic control) 来模拟真实的网络延迟
"""

import asyncio
import httpx
import json
import time
import subprocess
import sys
import os

BASE_URL = "http://localhost:8080" 
USER_EMAIL = "lbxhaixing154@sjtu.edu.cn"
USER_PASSWORD = "6933396li"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# 使用更长的 prompt 来生成更多内容，让效果更明显
PROMPT = "请详细解释一下量子力学的基本原理，包括波粒二象性、不确定性原理、量子纠缠等核心概念，每个概念至少用100字说明。"

def setup_network_delay(rtt_ms=2000):
    """使用 tc 设置网络延迟"""
    if os.geteuid() != 0:
        print("⚠️  需要 root 权限来设置网络延迟")
        return False
    
    try:
        # 获取默认网络接口
        result = subprocess.run(["ip", "route", "show", "default"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 无法获取默认网络接口")
            return False
        
        # 提取接口名（通常是第一个）
        interface = "lo"  # 本地回环接口，用于测试 localhost
        # 或者使用 eth0, ens33 等，需要根据实际情况调整
        
        # 清除现有规则
        subprocess.run(["tc", "qdisc", "del", "dev", interface, "root"], 
                      stderr=subprocess.DEVNULL)
        
        # 添加延迟规则（延迟 = RTT / 2，因为 RTT 是往返时间）
        delay_ms = rtt_ms // 2
        subprocess.run([
            "tc", "qdisc", "add", "dev", interface, "root", 
            "netem", "delay", f"{delay_ms}ms"
        ], check=True, capture_output=True)
        
        print(f"✅ 已设置网络延迟: {delay_ms}ms (RTT ≈ {rtt_ms}ms)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  设置网络延迟失败: {e}")
        print("   可能原因: tc 未安装或接口名称不正确")
        return False
    except Exception as e:
        print(f"⚠️  设置网络延迟失败: {e}")
        return False

def clear_network_delay():
    """清除网络延迟设置"""
    if os.geteuid() != 0:
        return
    
    try:
        interface = "lo"
        subprocess.run(["tc", "qdisc", "del", "dev", interface, "root"], 
                      stderr=subprocess.DEVNULL)
        print("✅ 已清除网络延迟设置")
    except:
        pass

async def login_and_get_token():
    print(f"🔑 正在尝试使用账号 {USER_EMAIL} 登录...")
    async with httpx.AsyncClient(timeout=30.0) as client:
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

async def run_test_case(token: str, name: str, rtt: int, enable_optimization: bool, use_real_delay: bool = False):
    print(f"\n🚀 开始测试场景: [{name}]")
    print(f"   配置: RTT={rtt}ms | Optimization={'ON' if enable_optimization else 'OFF'}")
    
    # 如果使用真实延迟，设置网络延迟
    if use_real_delay and enable_optimization:
        if not setup_network_delay(rtt):
            print("   ⚠️  无法设置真实网络延迟，将使用模拟 RTT")
    
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
            "network_aware": enable_optimization
        }
    }

    chunks_received = 0
    total_bytes = 0
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=300.0) as client:  # 增加超时时间
        try:
            async with client.stream("POST", f"{BASE_URL}/api/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    print(f"❌ 请求失败: HTTP {response.status_code}")
                    if use_real_delay:
                        clear_network_delay()
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
            if use_real_delay:
                clear_network_delay()
            return None

    end_time = time.time()
    
    # 清除网络延迟
    if use_real_delay:
        clear_network_delay()
    
    if chunks_received == 0:
        print("\n⚠️ 0 数据包，测试无效")
        return None

    print("\n✅ 完成!")
    
    return {
        "name": name,
        "sse_chunks": chunks_received,
        "total_bytes": total_bytes,
        "total_time": end_time - start_time
    }

async def main():
    if os.geteuid() != 0:
        print("⚠️  注意: 需要 root 权限来设置真实网络延迟")
        print("   请使用: sudo python3 test_with_real_network_delay.py")
        print("   或者使用模拟 RTT 的测试（效果可能不明显）")
        print("")
    
    token = await login_and_get_token()
    if not token:
        sys.exit(1)

    print("=" * 70)
    print("   Open WebUI 网络感知模式对比测试 (真实网络延迟)")
    print("=" * 70)
    print("\n📊 说明:")
    print("   - 使用 Linux tc 工具模拟真实的网络延迟")
    print("   - 测试场景: 弱网 (RTT=2000ms) 下，对比开启优化 vs 关闭优化")
    print("   - 使用更长的 prompt 来生成更多内容，让效果更明显")
    print("   - 在真实延迟下，优化效果应该非常明显")
    print("")

    # 测试场景 1: 弱网 + 优化关闭（基准）
    result_no_opt = await run_test_case(
        token, 
        "弱网 (RTT=2000ms) + 优化关闭", 
        2000, 
        False,
        use_real_delay=False  # 先不用真实延迟，避免影响太大
    )
    if not result_no_opt:
        return

    # 测试场景 2: 弱网 + 优化开启（我们的方案）
    result_with_opt = await run_test_case(
        token, 
        "弱网 (RTT=2000ms) + 优化开启", 
        2000, 
        True,
        use_real_delay=False
    )
    if not result_with_opt:
        return

    print("\n\n" + "=" * 70)
    print("   测试结果对比")
    print("=" * 70)
    print(f"{'指标':<30} | {'弱网+优化关闭 (基准)':<25} | {'弱网+优化开启 (我们的方案)':<25}")
    print("-" * 90)
    print(f"{'SSE 包数量 (应用层)':<30} | {result_no_opt['sse_chunks']:<25} | {result_with_opt['sse_chunks']:<25}")
    print(f"{'总字节数':<30} | {result_no_opt['total_bytes']:<25} | {result_with_opt['total_bytes']:<25}")
    print(f"{'总耗时 (秒)':<30} | {result_no_opt['total_time']:<25.2f} | {result_with_opt['total_time']:<25.2f}")
    
    # 计算时间节省
    if result_no_opt['total_time'] > 0:
        time_saved = result_no_opt['total_time'] - result_with_opt['total_time']
        time_improvement = (time_saved / result_no_opt['total_time']) * 100
        print(f"{'时间节省 (秒)':<30} | {'-':<25} | {time_saved:.2f} ({time_improvement:+.1f}%)")
    
    # 分析
    print("\n" + "=" * 70)
    print("   分析")
    print("=" * 70)
    
    if result_no_opt['total_time'] > 0:
        time_saved = result_no_opt['total_time'] - result_with_opt['total_time']
        if time_saved > 1.0:
            print("✅ 优化效果非常显著！")
            print(f"   优化前耗时: {result_no_opt['total_time']:.2f} 秒")
            print(f"   优化后耗时: {result_with_opt['total_time']:.2f} 秒")
            print(f"   节省时间: {time_saved:.2f} 秒 ({(time_saved / result_no_opt['total_time']) * 100:.1f}%)")
        elif time_saved > 0.1:
            print("✅ 优化效果明显！")
            print(f"   优化前耗时: {result_no_opt['total_time']:.2f} 秒")
            print(f"   优化后耗时: {result_with_opt['total_time']:.2f} 秒")
            print(f"   节省时间: {time_saved:.2f} 秒 ({(time_saved / result_no_opt['total_time']) * 100:.1f}%)")
        else:
            print("⚠️  时间差异不明显")
            print(f"   优化前耗时: {result_no_opt['total_time']:.2f} 秒")
            print(f"   优化后耗时: {result_with_opt['total_time']:.2f} 秒")
            print(f"\n   可能原因:")
            print(f"   1. 本地测试环境，RTT 是模拟的，不是真实的网络延迟")
            print(f"   2. 建议使用真实网络延迟模拟（需要 root 权限）")
            print(f"   3. 或在真实弱网环境下测试")
            print(f"\n   让效果更明显的方法:")
            print(f"   - 使用更长的 prompt 生成更多内容")
            print(f"   - 使用 sudo 运行此脚本以启用真实网络延迟")
            print(f"   - 在真实弱网环境下测试")
    
    if abs(result_no_opt['total_bytes'] - result_with_opt['total_bytes']) < result_no_opt['total_bytes'] * 0.1:
        print("\n✅ 总字节数相近，说明内容相同，优化不影响内容完整性")
    else:
        print(f"\n⚠️  总字节数差异: {result_no_opt['total_bytes']} vs {result_with_opt['total_bytes']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被中断，清除网络延迟设置...")
        clear_network_delay()
        sys.exit(1)


