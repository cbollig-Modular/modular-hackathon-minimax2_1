# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Mixture of Experts Layer for MiniMax-M2.1."""

from __future__ import annotations

from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.legacy.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.legacy.moe import MoE, MoEGate


class MiniMaxM2MoE(MoE):
    """MoE implementation for MiniMax-M2.1 with 256 experts and 8 per token.

    Uses stacked expert weights for efficient grouped matmul operations.
    Implements standard gated activation: gate * silu(up).
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate],
        dtype: DType = DType.bfloat16,
    ) -> None:
        """Initialize MiniMax-M2 MoE layer.

        Args:
            devices: Devices to place the module on.
            hidden_dim: Dimension of the hidden state.
            num_experts: Total number of experts (256 for MiniMax-M2).
            num_experts_per_token: Number of experts per token (8 for MiniMax-M2).
            moe_dim: Intermediate dimension of each expert (1536 for MiniMax-M2).
            gate_cls: Class to use for the MoE gate/router.
            dtype: Data type for the module weights.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=gate_cls,
            dtype=dtype,
            has_shared_experts=False,
            shared_experts_dim=0,
            ep_size=1,
            apply_router_weight_first=False,
            ep_batch_manager=None,
            float8_config=None,
            is_sharding=False,
        )

    def _init_experts(self) -> None:
        """Initialize experts using stacked weight tensors.

        This matches the HuggingFace checkpoint structure:
        - experts.gate_up_proj: [num_experts, hidden_dim, 2*moe_dim]
        - experts.down_proj: [num_experts, moe_dim, hidden_dim]

        Gate and up projections are combined into a single tensor for
        efficiency. This enables fused operations in grouped_matmul_ragged.
        """
        # Stacked gate and up projection weights
        # Shape: [num_experts, hidden_dim, 2*moe_dim]
        self._experts_gate_up_proj_weight = Weight(
            "experts.gate_up_proj",
            shape=[self.num_experts, self.hidden_dim, 2 * self.moe_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

        # Stacked down projection weights
        # Shape: [num_experts, moe_dim, hidden_dim]
        self._experts_down_proj_weight = Weight(
            "experts.down_proj",
            shape=[self.num_experts, self.moe_dim, self.hidden_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        """Return gate_up projection weights for grouped_matmul_ragged.

        Returns:
            Weight tensor with shape [num_experts, 2*moe_dim, hidden_dim].
            Transposed to match grouped_matmul_ragged expected format.
        """
        # grouped_matmul_ragged expects [num_experts, out_features, in_features]
        return self._experts_gate_up_proj_weight.transpose(1, 2)

    @property
    def down_proj(self) -> TensorValue:
        """Return down projection weights for grouped_matmul_ragged.

        Returns:
            Weight tensor with shape [num_experts, hidden_dim, moe_dim].
            Transposed to match grouped_matmul_ragged expected format.
        """
        # grouped_matmul_ragged expects [num_experts, out_features, in_features]
        return self._experts_down_proj_weight.transpose(1, 2)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module.

        Args:
            strategy: The sharding strategy to apply.

        Raises:
            ValueError: If strategy is not tensor parallel.
        """
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy
            # Gate is replicated across devices
            self.gate.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            # Expert weights are sharded
            self._experts_gate_up_proj_weight.sharding_strategy = (
                ShardingStrategy.gate_up(strategy.num_devices)
            )
            self._experts_down_proj_weight.sharding_strategy = (
                ShardingStrategy.axiswise(
                    axis=1, num_devices=strategy.num_devices
                )
            )
        else:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for MiniMaxM2MoE"
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape (seq_len, hidden_dim).

        Returns:
            Output tensor of shape (seq_len, hidden_dim).
        """
        # Route tokens to experts
        router_idx, router_weight = self.gate(x)

        # Flatten router indices: (seq_len, num_experts_per_token) -> (seq_len * num_experts_per_token,)
        router_idx = ops.reshape(router_idx, [-1])
        router_idx_int32 = ops.cast(router_idx, DType.int32)

        # Create routing indices for expert assignment
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(router_idx_int32, self.num_experts)

        # Extract token indices from token_expert_order
        # token_expert_order has indices into flattened router_idx array
        token_indices = ops.cast(
            token_expert_order // self.num_experts_per_token, DType.int32
        )

        # Permute tokens according to expert assignment
        permutated_states = ops.gather(x, token_indices, axis=0)

        # Gate + Up Projection via grouped matmul
        # Input: (num_tokens, hidden_dim)
        # Output: (num_tokens, 2*moe_dim)
        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        # Split into gate and up projections
        up = gate_up_projs[:, self.moe_dim :]
        gate = gate_up_projs[:, : self.moe_dim]

        # Apply gated activation: gate * silu(up)
        gate_up_projs = up * ops.silu(gate)

        # Down Projection via grouped matmul
        # Input: (num_tokens, moe_dim)
        # Output: (num_tokens, hidden_dim)
        expert_outputs = grouped_matmul_ragged(
            gate_up_projs,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        # Restore original token order and apply router weights
        # router_weight shape: (seq_len, num_experts_per_token)
        router_weight_flat = ops.reshape(
            router_weight, [-1, 1]
        )  # (seq_len * num_experts_per_token, 1)
        weighted_outputs = expert_outputs * router_weight_flat

        # Gather back to original token order
        # restore_token_order maps from permuted order back to original
        restore_token_order_int32 = ops.cast(restore_token_order, DType.int32)
        ordered_weighted_outputs = ops.gather(
            weighted_outputs, restore_token_order_int32, axis=0
        )

        # Reshape to (seq_len, num_experts_per_token, hidden_dim)
        seq_len = x.shape[0]
        ordered_weighted_outputs = ops.reshape(
            ordered_weighted_outputs,
            [seq_len, self.num_experts_per_token, self.hidden_dim],
        )

        # Sum over experts: (seq_len, num_experts_per_token, hidden_dim) -> (seq_len, hidden_dim)
        final_outputs = ops.sum(ordered_weighted_outputs, axis=1)

        return final_outputs
