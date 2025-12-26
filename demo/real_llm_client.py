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


def main():
    parser = argparse.ArgumentParser(
        description="Real LLM Client with Network-Aware Token Pacing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use vLLM with default settings
  python real_llm_client.py --engine vllm --prompt "Hello, world!"
  
  # Use Ollama with custom model
  python real_llm_client.py --engine ollama --ollama-model llama2 --prompt "Tell me a story"
  
  # Custom Hint Server URL
  python real_llm_client.py --engine vllm --hint-url http://localhost:5000/hint --prompt "Hello"
        """
    )
    
    # Engine selection
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "ollama"],
        required=True,
        help="LLM inference engine to use"
    )
    
    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for the LLM"
    )
    
    # Hint Server configuration
    parser.add_argument(
        "--hint-url",
        type=str,
        default="http://localhost:5000/hint",
        help="Hint Server URL (default: http://localhost:5000/hint)"
    )
    
    # vLLM configuration
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default="default",
        help="vLLM model name (default: 'default')"
    )
    
    # Ollama configuration
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama2",
        help="Ollama model name (default: llama2)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    # Rate limiting
    parser.add_argument(
        "--disable-rate-limit",
        action="store_true",
        help="Disable network-aware rate limiting (for testing)"
    )
    
    args = parser.parse_args()
    
    # Initialize scheduler
    if args.disable_rate_limit:
        scheduler = None
        print("⚠️  Rate limiting disabled (testing mode)")
    else:
        print(f"🔗 Connecting to Hint Server: {args.hint_url}")
        scheduler = NetworkAwareScheduler(hint_url=args.hint_url)
        time.sleep(1)  # Wait for initial fetch
        print(f"✅ Initial rate: {scheduler.get_rate():.1f} tps, Health: {scheduler.get_health():.2f}")
    
    # Initialize LLM client
    if args.engine == "vllm":
        print(f"🚀 Using vLLM at {args.vllm_url}")
        client = VLLMClient(base_url=args.vllm_url)
        try:
            generator = client.generate_stream(
                prompt=args.prompt,
                model=args.vllm_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                scheduler=scheduler,
                hint_url=args.hint_url
            )
        except RuntimeError as e:
            print(f"❌ Error: {e}")
            print("   Make sure vLLM server is running and accessible.")
            sys.exit(1)
    
    elif args.engine == "ollama":
        print(f"🚀 Using Ollama at {args.ollama_url} with model '{args.ollama_model}'")
        client = OllamaClient(base_url=args.ollama_url)
        try:
            generator = client.generate_stream(
                prompt=args.prompt,
                model=args.ollama_model,
                max_tokens=args.max_tokens if args.max_tokens > 0 else None,
                temperature=args.temperature,
                scheduler=scheduler
                # Ollama client implementation in this file doesn't support hint_url yet in signature
                # We would need to update OllamaClient.generate_stream to accept hint_url as well if needed
            )
        except RuntimeError as e:
            print(f"❌ Error: {e}")
            print("   Make sure Ollama is running and the model is available.")
            print(f"   You can pull a model with: ollama pull {args.ollama_model}")
            sys.exit(1)
    
    # Generate and print tokens
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"🤖 Response (server-controlled rate):\n")
    print("─" * 60)
    
    start_time = time.time()
    token_count = 0
    
    try:
        for token in generator:
            sys.stdout.write(token)
            sys.stdout.flush()
            token_count += 1
            
            # Show rate info periodically
            if scheduler and token_count % 10 == 0:
                rate = scheduler.get_rate()
                health = scheduler.get_health()
                metrics = scheduler.get_metrics()
                sys.stdout.write(f"\n[Rate: {rate:.1f} tps, Health: {health:.2f}] ")
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    duration = time.time() - start_time
    actual_rate = token_count / duration if duration > 0 else 0
    
    print("\n" + "─" * 60)
    print(f"\n✅ Generated {token_count} tokens in {duration:.2f}s")
    print(f"   Actual rate: {actual_rate:.1f} tps")
    if scheduler:
        print(f"   Target rate: {scheduler.get_rate():.1f} tps")
        print(f"   Network health: {scheduler.get_health():.2f}")
        metrics = scheduler.get_metrics()
        if metrics:
            print(f"   Network metrics: RTT={metrics.get('rtt', 'N/A')}us, "
                  f"Retrans={metrics.get('retrans', 'N/A')}")
    
    if scheduler:
        scheduler.stop()


if __name__ == "__main__":
    main()

