# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence and its related classes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

import time
import torch

if TYPE_CHECKING:
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
else:
    KVConnectorOutput = Any

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """

    arrival_time: float
    last_token_time: float
    first_scheduled_time: float | None
    first_token_time: float | None
    time_in_queue: float | None
    finished_time: float | None = None
    scheduler_time: float | None = None
    model_forward_time: float | None = None
    model_execute_time: float | None = None


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.

    Each stage also needs to handle its own kv_connector_output.
    """

    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors
        self.kv_connector_output = kv_connector_output

    def __getitem__(self, key: str | slice):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            return False
        if self.tensors.keys() != other.tensors.keys():
            return False
        return all(torch.equal(self.tensors[k], other.tensors[k]) for k in self.tensors)

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


class Sequence:
    """Represents a single sequence in a SequenceGroup.
    
    This is a minimal placeholder class for compatibility.
    In a full implementation, this would contain sequence-specific data.
    """
    def __init__(self, seq_id: str):
        self.seq_id = seq_id


class SequenceGroup:
    """
    A group of sequences that share the same prompt and KV cache blocks.
    
    This class has been extended with network-aware scheduling capabilities
    to support dynamic rate control based on network conditions.
    """
    
    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
        # ... (保留现有参数，如果有的话)
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.arrival_time = arrival_time
        
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION START] ---
        # Control Signals (由 API 设置)
        self.target_qps: float = -1.0  # R_net: 目标网络速率
        self.dynamic_weight: float = 1.0  # W: 动态调度权重
        
        # Instrumentation / Metrics (用于计算 R_gpu)
        self.total_tokens_generated: int = 0
        self.gpu_execution_start_time: float = time.time()
        self.last_stats_update_time: float = time.time()
        self.active_time_cumulative: float = 0.0  # 累计有效计算时间
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION END] ---
    
    def get_actual_gpu_rate(self) -> float:
        """
        [NEW METHOD]
        Calculates the actual generation rate (Tokens Per Second) 
        based on active execution time.
        
        Formula: R_gpu = total_tokens_generated / max(active_time_cumulative, 0.001)
        
        Returns:
            float: The actual GPU generation rate in tokens per second.
                  Returns 0.0 if active_time_cumulative is too small.
        """
        if self.active_time_cumulative <= 0.001:
            return 0.0
        
        return self.total_tokens_generated / self.active_time_cumulative
    
    def update_execution_stats(self, new_tokens: int, time_delta: float):
        """
        [NEW METHOD]
        Called by the scheduler after a step is executed.
        Updates the cumulative statistics for rate calculation.
        
        Args:
            new_tokens: Number of new tokens generated in this step
            time_delta: Time spent in active execution (in seconds)
        """
        self.total_tokens_generated += new_tokens
        self.active_time_cumulative += time_delta
        self.last_stats_update_time = time.time()
