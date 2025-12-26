#!/usr/bin/env python3
"""
Simple test script to verify SequenceGroup modifications without full vllm import.
"""
import sys
import time
import importlib.util

# Load sequence.py directly without importing the full vllm package
spec = importlib.util.spec_from_file_location(
    "sequence", 
    "vllm/vllm/sequence.py"
)
sequence_module = importlib.util.module_from_spec(spec)

# Mock torch for testing
class MockTensor:
    pass

class MockTorch:
    Tensor = MockTensor
    def equal(self, a, b):
        return a == b

sys.modules['torch'] = MockTorch()
sys.modules['vllm'] = type(sys)('vllm')
sys.modules['vllm.v1'] = type(sys)('vllm.v1')
sys.modules['vllm.v1.worker'] = type(sys)('vllm.v1.worker')
sys.modules['vllm.v1.worker.kv_connector_model_runner_mixin'] = type(sys)('kv_connector')

spec.loader.exec_module(sequence_module)

SequenceGroup = sequence_module.SequenceGroup
Sequence = sequence_module.Sequence

def test_sequence_group():
    """Test SequenceGroup with network-aware scheduling fields."""
    print("🧪 Testing SequenceGroup network-aware scheduling...")
    
    # Create a SequenceGroup
    seqs = [Sequence("seq_0")]
    sg = SequenceGroup(
        request_id="req_001",
        seqs=seqs,
        arrival_time=time.time()
    )
    
    # Verify initial values
    print(f"✅ Created SequenceGroup: {sg.request_id}")
    print(f"   Initial target_qps: {sg.target_qps}")
    print(f"   Initial dynamic_weight: {sg.dynamic_weight}")
    print(f"   Initial total_tokens_generated: {sg.total_tokens_generated}")
    print(f"   Initial active_time_cumulative: {sg.active_time_cumulative}")
    
    # Test setting control parameters
    sg.target_qps = 50.0  # R_net
    sg.dynamic_weight = 0.8  # W
    print(f"\n✅ Set control parameters:")
    print(f"   target_qps: {sg.target_qps}")
    print(f"   dynamic_weight: {sg.dynamic_weight}")
    
    # Test update_execution_stats
    print(f"\n✅ Testing update_execution_stats...")
    sg.update_execution_stats(new_tokens=10, time_delta=0.2)
    print(f"   After 10 tokens in 0.2s:")
    print(f"   total_tokens_generated: {sg.total_tokens_generated}")
    print(f"   active_time_cumulative: {sg.active_time_cumulative:.3f}")
    
    sg.update_execution_stats(new_tokens=15, time_delta=0.3)
    print(f"   After 15 more tokens in 0.3s:")
    print(f"   total_tokens_generated: {sg.total_tokens_generated}")
    print(f"   active_time_cumulative: {sg.active_time_cumulative:.3f}")
    
    # Test get_actual_gpu_rate
    rate = sg.get_actual_gpu_rate()
    print(f"\n✅ Testing get_actual_gpu_rate...")
    print(f"   Actual GPU rate: {rate:.2f} tokens/second")
    expected = sg.total_tokens_generated / sg.active_time_cumulative
    print(f"   Expected: {expected:.2f} tokens/second")
    assert abs(rate - expected) < 0.01, f"Rate mismatch: {rate} vs {expected}"
    
    # Test edge case: zero active time
    sg2 = SequenceGroup("req_002", [Sequence("seq_1")], time.time())
    rate_zero = sg2.get_actual_gpu_rate()
    print(f"\n✅ Testing edge case (zero active time):")
    print(f"   Rate with zero active time: {rate_zero}")
    assert rate_zero == 0.0, "Rate should be 0.0 when active_time_cumulative is too small"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_sequence_group()

