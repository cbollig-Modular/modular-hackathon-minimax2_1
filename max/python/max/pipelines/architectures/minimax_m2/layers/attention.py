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

"""MiniMax-M2.1 Attention Layer with QK Normalization."""

from __future__ import annotations

import math
from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import Llama3RotaryEmbedding


class MiniMaxM2Attention(Module):
    """Attention layer for MiniMax-M2 with QK normalization.

    Implements grouped-query attention (GQA) with:
    - 48 query heads
    - 8 key-value heads
    - RoPE embeddings with theta=5,000,000
    - QK normalization for training stability
    - Paged KV cache for long context support
    """

    def __init__(
        self,
        *,
        rope: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.bfloat16,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        """Initialize the attention layer.

        Args:
            rope: Rotary position embedding.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA.
            hidden_size: Dimension of hidden states.
            kv_params: KV cache parameters.
            layer_idx: Layer index in the model.
            dtype: Data type for weights and activations.
            devices: Device to place the weights and run computation.
            linear_cls: Linear layer class to use.
            scale: Attention scale factor (default: 1/sqrt(head_dim)).
            has_bias: Whether to use bias in projections.
            qk_norm_eps: Epsilon for QK normalization.
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.qk_norm_eps = qk_norm_eps
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy not supported"
                " in MiniMaxM2Attention layer."
            )

        # QK normalization layers
        self.q_norm = RMSNorm(
            self.kv_params.head_dim, DType.bfloat16, self.qk_norm_eps
        )
        self.k_norm = RMSNorm(
            self.kv_params.head_dim, DType.bfloat16, self.qk_norm_eps
        )

        # Compute projection dimensions
        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        # Q, K, V projections
        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )

        # Output projection
        self.o_proj = linear_cls(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        """Concatenation of Q, K, V weight vectors."""
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        wqkv = ops.concat((wq, wk, wv))
        return wqkv.to(self.devices[0])

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """Concatenation of Q, K, V bias vectors."""
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        )

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        """Forward pass through the attention layer.

        Args:
            x: Input tensor of shape (total_seq_len, hidden_dim).
            kv_collection: Paged KV cache collection.
            **kwargs: Additional arguments including input_row_offsets.

        Returns:
            Attention output tensor of shape (total_seq_len, hidden_dim).
        """
        total_seq_len = x.shape[0]

        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        # Fused QKV projection and KV cache write
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Reshape queries to (total_seq_len, n_heads, head_dim)
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply QK normalization
        xq = self.q_norm(xq)
        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )

        # Apply rotary embeddings
        freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Flash attention
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        # Reshape and apply output projection
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
