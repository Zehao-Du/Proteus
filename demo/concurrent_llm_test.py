#!/usr/bin/env python3
"""
Real LLM Client with Network-Aware Token Pacing

This client integrates with real LLM inference engines (vLLM or Ollama) and
respects the rate limits provided by the Hint Server based on network conditions.

Supported Engines:
- vLLM: High-performance LLM inference engine
- Ollama: Local LLM runner

Usage:
    # With vLLM
    python real_llm_client.py --engine vllm --vllm-url http://localhost:8000/v1 --prompt "Hello, world!"

    # With Ollama
    python real_llm_client.py --engine ollama --ollama-model llama2 --prompt "Hello, world!"

    # Custom Hint Server
    python real_llm_client.py --engine ollama --hint-url http://localhost:5000/hint --prompt "Hello, world!"
"""
import argparse
import json
import sys
import threading
import time
from typing import Optional, Iterator, Dict, Any

import requests


class NetworkAwareScheduler:
    """Scheduler that queries Hint Server and enforces rate limits."""
    
    def __init__(self, hint_url: str = "http://localhost:5000/hint", poll_interval: float = 0.5):
        self.hint_url = hint_url
        self.poll_interval = poll_interval
        self.current_rate = 20.0  # Default fallback rate (tokens per second)
        self.health = 1.0
        self.metrics = {}
        self.running = True
        self._lock = threading.Lock()
        self._last_token_time = {}  # Track last token emission time per connection
        
        # Start background poller thread
        self.thread = threading.Thread(target=self._poller, daemon=True)
        self.thread.start()
        
        # Wait a bit for initial fetch
        time.sleep(0.5)
    
    def update_server_rate(self, new_rate: float, conn_id: str = "default"):
        """
        Simulates sending the recommended rate back to the LLM Server.
        """
        pass
    
    def _poller(self):
        """Background thread that periodically queries Hint Server and updates server."""
        while self.running:
            try:
                resp = requests.get(self.hint_url, timeout=0.5)
                if resp.status_code == 200:
                    data = resp.json()
                    new_rate = data.get("token_rate", 20.0)
                    
                    with self._lock:
                        # 1. Update internal state
                        self.current_rate = new_rate
                        self.health = data.get("health", 1.0)
                        self.metrics = data.get("metrics", {})
                    
                    # 2. Call new update method after lock is released
                    self.update_server_rate(new_rate, "default")
                    
            except Exception as e:
                # Silently fail and keep using last known rate
                pass
            time.sleep(self.poll_interval)
    
    def wait_for_slot(self, hint_url: str = "default"):
        """
        Wait until it's safe to emit the next token based on current rate limit.
        """
        now = time.time()
        
        with self._lock:
            # We treat hint_url as the connection ID for simplicity in this revised call
            conn_id = hint_url 
            if conn_id not in self._last_token_time:
                self._last_token_time[conn_id] = now
                return
            
            last_emit = self._last_token_time[conn_id]
            required_interval = 1.0 / max(self.current_rate, 0.1)  # Minimum 0.1 tps
            elapsed = now - last_emit
            wait_time = required_interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self._last_token_time[conn_id] = time.time()
    
    def get_rate(self) -> float:
        """Get current recommended token rate."""
        with self._lock:
            return self.current_rate
    
    def get_health(self) -> float:
        """Get current network health score (0.0 to 1.0)."""
        with self._lock:
            return self.health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current network metrics."""
        with self._lock:
            return self.metrics.copy()
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
    
    def get_rate(self) -> float:
        """Get current recommended token rate."""
        with self._lock:
            return self.current_rate
    
    def get_health(self) -> float:
        """Get current network health score (0.0 to 1.0)."""
        with self._lock:
            return self.health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current network metrics."""
        with self._lock:
            return self.metrics.copy()
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False


class VLLMClient:
    """Client for vLLM inference engine."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/chat/completions"
        self.completions_url = f"{self.base_url}/completions"
    
    def generate_stream(
        self, 
        prompt: str, 
        model: str = "default",
        max_tokens: int = 100,
        temperature: float = 0.7,
        scheduler: Optional[NetworkAwareScheduler] = None,
        hint_url: str = ""
    ) -> Iterator[str]:
        """
        Generate tokens from vLLM using the robust endpoints.
        """
        # Try to auto-detect model if "default" is provided
        if model == "default":
            try:
                models_resp = requests.get(f"{self.base_url}/models", timeout=5)
                if models_resp.status_code == 200:
                    data = models_resp.json()
                    if data.get("data"):
                        model = data["data"][0]["id"]
                        print(f"ℹ️  Auto-detected vLLM model: {model}")
            except Exception as e:
                print(f"⚠️  Failed to auto-detect model: {e}")

        # Use standard Chat Completions API
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        try:
            # Use standard chat completions endpoint
            response = requests.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                if line.startswith(b"data: "):
                    data_str = line[6:].decode('utf-8')
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if scheduler and hint_url:
                                    scheduler.wait_for_slot(hint_url)
                                yield content
                    except json.JSONDecodeError:
                        continue
        
        except requests.exceptions.RequestException as e:
            # Fallback to legacy completions endpoint
            payload_raw = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True
            }
            try:
                response = requests.post(f"{self.base_url}/completions", json=payload_raw, stream=True)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        data_str = line[6:].decode('utf-8')
                        if data_str == "[DONE]": break
                        data = json.loads(data_str)
                        # v1/completions logic
                        text = data.get("text", "")
                        if text:
                            if scheduler and hint_url: scheduler.wait_for_slot(hint_url)
                            yield text
                return
            except:
                pass
            raise RuntimeError(f"vLLM request failed: {e}")


class OllamaClient:
    """Client for Ollama inference engine."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
    
    def generate_stream(
        self,
        prompt: str,
        model: str = "llama2",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        scheduler: Optional[NetworkAwareScheduler] = None
    ) -> Iterator[str]:
        """
        Generate tokens from Ollama with streaming.
        Rate limiting is now handled by the server based on network conditions.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate (None = no limit)
            temperature: Sampling temperature
            scheduler: Network-aware scheduler (for monitoring, not rate limiting)
        
        Yields:
            Token strings as they are generated
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            conn_id = "ollama_stream"
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        # --- REMOVED CLIENT-SIDE RATE LIMITING ---
                        # Server now controls the generation rate internally
                        # Client receives and displays tokens as fast as they arrive
                        yield token
                    
                    # Check if done
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")


import concurrent.futures
import random

def run_single_client(
    client_id: int, 
    engine: str, 
    prompt: str, 
    hint_url: str, 
    vllm_url: str,
    vllm_model: str,
    max_tokens: int,
    temperature: float,
    disable_rate_limit: bool
):
    prefix = f"[User {client_id}] "
    print(f"{prefix}🚀 Starting request...")
    
    # Each user gets their own scheduler to simulate their own network condition
    # In a real scenario, this might be handled by a single scheduler with per-connection tracking
    # For this simulation, we'll let them share the Hint Server query logic but we'll print distinctly
    
    scheduler = None
    if not disable_rate_limit:
        # To simulate different network conditions, we could pass different hint_urls or 
        # modify the scheduler to support client IDs. 
        # For this demo, we use the same hint server but they will see the same global "health"
        # unless we modify the Hint Server to return different healths per IP.
        # But wait! Our thesis is: "If User A is slow, User B gets more."
        # The vLLM server is the one "allocating" the budget. 
        # So we just need to see if User B (who might be "lucky" in this simulation or just concurrent) gets good throughput.
        scheduler = NetworkAwareScheduler(hint_url=hint_url)
        time.sleep(random.uniform(0.1, 0.5)) # Jitter

    token_count = 0
    start_time = time.time()
    
    try:
        if engine == "vllm":
            client = VLLMClient(base_url=vllm_url)
            generator = client.generate_stream(
                prompt=prompt,
                model=vllm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                scheduler=scheduler,
                hint_url=hint_url
            )
        else:
             # Ollama support skipped for concurrency test brevity
             print(f"{prefix}❌ Ollama not supported in concurrent test yet")
             return

        for token in generator:
            token_count += 1
            # Don't print every token to avoid console chaos, just progress dots
            if token_count % 10 == 0:
                 sys.stdout.write(f".")
                 sys.stdout.flush()
    except Exception as e:
        print(f"\n{prefix}❌ Error: {e}")
        return

    duration = time.time() - start_time
    actual_rate = token_count / duration if duration > 0 else 0
    print(f"\n{prefix}✅ Done! {token_count} tokens in {duration:.2f}s (Rate: {actual_rate:.1f} tps)")


def main():
    parser = argparse.ArgumentParser(description="Concurrent LLM Client Test")
    # ... (keep existing args setup but simplify for this test) ...
    parser.add_argument("--num-clients", type=int, default=3, help="Number of concurrent clients")
    
    # Reuse existing args
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--prompt", type=str, default="Write a short poem about future cities.")
    parser.add_argument("--hint-url", type=str, default="http://localhost:5000/hint")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", type=str, default="default")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--disable-rate-limit", action="store_true")

    args = parser.parse_args()

    print(f"🔥 Starting {args.num_clients} concurrent clients...")
    print("-" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_clients) as executor:
        futures = []
        for i in range(args.num_clients):
            # We can vary prompts or keep them same
            futures.append(
                executor.submit(
                    run_single_client, 
                    i+1, 
                    args.engine, 
                    args.prompt, 
                    args.hint_url, 
                    args.vllm_url, 
                    args.vllm_model, 
                    args.max_tokens, 
                    args.temperature, 
                    args.disable_rate_limit
                )
            )
        
        concurrent.futures.wait(futures)
    
    print("-" * 60)
    print("🏁 All clients finished.")

if __name__ == "__main__":
    main()

