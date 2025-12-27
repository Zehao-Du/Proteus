# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of requests being preempted by the scheduler
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()
        
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION START] ---
        # Per-request health factor (0.0 - 1.0)
        # 每个请求独立的健康度，用于 per-user 算力分配
        self.health_factor: float = 1.0
        
        # Control Signals (由 API 设置)
        self.target_qps: float = -1.0  # R_net: 目标网络速率
        self.dynamic_weight: float = 1.0  # W: 动态调度权重
        
        # Instrumentation / Metrics (用于计算 R_gpu)
        self.total_tokens_generated: int = 0
        self.gpu_execution_start_time: float = time.time()
        self.last_stats_update_time: float = time.time()
        self.active_time_cumulative: float = 0.0  # 累计有效计算时间
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION END] ---

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        req = cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )
        
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION START] ---
        # 从 extra_args 中提取 health_factor（如果提供）
        # 这样客户端可以直接传递健康度，避免查询 hint server
        if request.sampling_params and hasattr(request.sampling_params, 'extra_args'):
            extra_args = request.sampling_params.extra_args
            if extra_args and isinstance(extra_args, dict):
                health_factor = extra_args.get("health_factor")
                if health_factor is not None:
                    try:
                        req.health_factor = float(health_factor)
                    except (ValueError, TypeError):
                        pass  # 使用默认值 1.0
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION END] ---
        
        return req

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        num_tokens = self.mm_features[input_id].mm_position.length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

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
    
    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, health_factor, and arrival time.
        Used in priority scheduling.
        
        [NETWORK-AWARE SCHEDULING MODIFICATION]
        Modified to incorporate health_factor for network-aware scheduling.
        Higher health_factor = better network = higher priority.
        """
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION START] ---
        # First compare by priority (existing behavior)
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # Then compare by health_factor (higher health = higher priority)
        # Higher health = smaller value in heap = scheduled first
        if abs(self.health_factor - other.health_factor) > 0.01:
            return self.health_factor > other.health_factor  # 高健康度优先
        # --- [NETWORK-AWARE SCHEDULING MODIFICATION END] ---
        
        # Finally compare by arrival time (FCFS within same priority and health)
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
}
