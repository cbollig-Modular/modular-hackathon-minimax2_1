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

"""Mixture of Experts Gate Layer for MiniMax-M2.1."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.legacy.kernels import moe_router_group_limited
from max.nn.legacy.linear import Linear
from max.nn.legacy.moe import MoEGate
from max.nn.legacy.moe.moe import ShardingStrategy


class MiniMaxM2TopKRouter(MoEGate):
    """Mixture of Experts Gate Layer for MiniMax-M2.

    This router uses sigmoid-based scoring with a correction bias to select
    the top-k experts for each token. The routing is group-limited to ensure
    load balancing across experts.
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        routed_scaling_factor: float = 1.0,
        n_group: int | None = None,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        correction_bias_dtype: DType | None = None,
        linear_cls: Callable[..., Linear] = Linear,
    ) -> None:
        """Initialize the MiniMax-M2 MoE router.

        Args:
            devices: Devices to place the module on.
            hidden_dim: Dimension of the hidden state.
            num_experts: Total number of experts in the MoE layer.
            num_experts_per_token: Number of experts to route each token to.
            dtype: Data type for the gate scoring layer.
            routed_scaling_factor: Scaling factor applied to routed expert outputs.
            n_group: Number of expert groups for load balancing. If None, defaults to num_experts // num_experts_per_token.
            topk_group: Number of experts to select from each group.
            norm_topk_prob: Whether to normalize the top-k routing probabilities.
            correction_bias_dtype: Data type for the correction bias. If None, uses float32.
            linear_cls: Linear layer class to use for gate scoring.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            linear_cls=linear_cls,
        )

        # Set defaults
        if n_group is None:
            n_group = num_experts // num_experts_per_token
        if correction_bias_dtype is None:
            correction_bias_dtype = DType.float32

        self.top_k = num_experts_per_token
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.correction_bias_dtype = correction_bias_dtype

        if self.num_experts % self.n_group != 0:
            raise ValueError(
                f"num_experts must be divisible by n_group: "
                f"{self.num_experts} % {self.n_group} != 0"
            )

        # Correction bias for expert routing
        # This bias is added to expert scores to improve routing decisions
        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[self.num_experts],
            device=self.devices[0],
            dtype=correction_bias_dtype,
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim).

        Returns:
            Tuple containing:
                - topk_idx: Indices of top-k selected experts, shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts, shape (seq_len, num_experts_per_token)
        """
        # Compute gate logits
        logits = self.gate_score(hidden_states)

        # Apply sigmoid activation and cast to correction bias dtype
        scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))

        # Select top-k experts using group-limited routing
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return topk_idx, topk_weight

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
            ValueError: If strategy is not replicate (only replicate is supported).
        """
        if strategy.is_replicate:
            self._sharding_strategy = strategy
            self.gate_score.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.e_score_correction_bias.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
        else:
            raise ValueError(
                "Only replicate sharding strategy is supported for MoEGate."
            )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MiniMaxM2TopKRouter instances, one for each device.

        Raises:
            ValueError: If no sharding strategy is set.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = MiniMaxM2TopKRouter(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                routed_scaling_factor=self.routed_scaling_factor,
                n_group=self.n_group,
                topk_group=self.topk_group,
                norm_topk_prob=self.norm_topk_prob,
                correction_bias_dtype=self.correction_bias_dtype,
            )

            # Replace the weights with sharded versions
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
