#!/usr/bin/env python3
import time
import sys
import threading
import requests
import asyncio

class NetworkAwareScheduler:
    def __init__(self, hint_url="http://localhost:5000/hint"):
        self.hint_url = hint_url
        self.current_rate = 20.0
        self.running = True
        self.thread = threading.Thread(target=self._poller, daemon=True)
        self.thread.start()
        self.conn_states = {}

    def _poller(self):
        while self.running:
            try:
                resp = requests.get(self.hint_url, timeout=0.2)
                if resp.status_code == 200:
                    data = resp.json()
                    self.current_rate = data.get("token_rate", 20.0)
            except:
                pass
            time.sleep(0.5)

    def wait_for_slot(self, conn_id):
        now = time.time()
        if conn_id not in self.conn_states:
            self.conn_states[conn_id] = now
            return
        
        last_emit = self.conn_states[conn_id]
        required_interval = 1.0 / max(self.current_rate, 0.1)
        elapsed = now - last_emit
        wait = required_interval - elapsed
        
        if wait > 0:
            time.sleep(wait)
        
        self.conn_states[conn_id] = time.time()

    def get_rate(self):
        return self.current_rate

async def mock_generate(scheduler):
    conn_id = "test_conn"
    total = 500
    print(f"🤖 Starting mock LLM generation ({total} tokens)...")
    
    start = time.time()
    for i in range(total):
        scheduler.wait_for_slot(conn_id)
        rate = scheduler.get_rate()
        sys.stdout.write(f"\rTok {i+1}/{total} | Rate: {rate:.1f} tps | {'#' * int((i/total)*20)}")
        sys.stdout.flush()
    
    duration = time.time() - start
    print(f"\n✅ Done. Avg Speed: {total/duration:.1f} tps")

if __name__ == "__main__":
    scheduler = NetworkAwareScheduler()
    print("⏳ Waiting for sync...")
    time.sleep(2)
    asyncio.run(mock_generate(scheduler))

