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

"""MiniMax-M2.1 decoder model implementation."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import Llama3RotaryEmbedding

from .layers import MiniMaxM2Attention, MiniMaxM2MoE, MiniMaxM2TopKRouter
from .model_config import MiniMaxM2Config


class MiniMaxM2DecoderLayer(Module):
    """A single decoder layer for MiniMax-M2.1.

    Consists of:
    - Pre-attention RMS normalization
    - Self-attention with QK normalization
    - Post-attention RMS normalization (with residual)
    - Pre-MoE RMS normalization
    - MoE layer with 256 experts
    - Post-MoE residual connection
    """

    def __init__(
        self,
        *,
        attention: MiniMaxM2Attention,
        moe: MiniMaxM2MoE,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> None:
        """Initialize a decoder layer.

        Args:
            attention: Self-attention module.
            moe: Mixture-of-experts module.
            input_layernorm: Pre-attention normalization.
            post_attention_layernorm: Pre-MoE normalization.
        """
        super().__init__()
        self.self_attn = attention
        self.mlp = moe
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def __call__(
        self,
        hidden_states: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim).
            kv_collection: Paged KV cache collection.
            **kwargs: Additional arguments for attention (e.g., input_row_offsets).

        Returns:
            Output tensor of shape (seq_len, hidden_dim).
        """
        # Self-attention block with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, kv_collection=kv_collection, **kwargs
        )
        hidden_states = residual + hidden_states

        # MoE block with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MiniMaxM2(Module):
    """The MiniMax-M2.1 transformer decoder model.

    Consists of:
    - Token embedding layer
    - 62 decoder layers with attention + MoE
    - Final RMS normalization
    - Language model head
    """

    def __init__(self, config: MiniMaxM2Config) -> None:
        """Initialize the MiniMax-M2 model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.devices = config.devices
        self.vocab_size = config.vocab_size
        self.config = config

        # Rotary position embeddings
        self.rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        # Token embeddings
        self.embed_tokens = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        # Create RMSNorm factory function
        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Create decoder layers
        layers = [
            MiniMaxM2DecoderLayer(
                attention=MiniMaxM2Attention(
                    rope=self.rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    has_bias=config.attention_bias,
                    qk_norm_eps=config.rms_norm_eps,
                ),
                moe=MiniMaxM2MoE(
                    devices=config.devices,
                    hidden_dim=config.hidden_size,
                    num_experts=config.num_local_experts,
                    num_experts_per_token=config.num_experts_per_tok,
                    moe_dim=config.intermediate_size,
                    gate_cls=MiniMaxM2TopKRouter,
                    dtype=config.dtype,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]

        self.layers = LayerList(layers)

        # Final normalization
        self.norm = create_norm()

        # Language model head
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        kv_collection: PagedCacheValues,
    ) -> TensorValue:
        """Forward pass through the model.

        Args:
            tokens: Input token IDs of shape (total_tokens,).
            input_row_offsets: Row offsets for ragged batching.
            kv_collection: Paged KV cache collection.

        Returns:
            Logits tensor of shape (total_tokens, vocab_size) or (total_tokens, 1)
            depending on return_logits configuration.
        """
        # Embed tokens
        hidden_states = self.embed_tokens(tokens)

        # Pass through all decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                kv_collection=kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Language model head
        logits = self.lm_head(hidden_states)

        return logits
