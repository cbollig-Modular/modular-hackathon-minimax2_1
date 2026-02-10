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

"""MiniMax-M2.1 model configuration for MAX pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class MiniMaxM2Config(ArchConfigWithKVCache):
    """Configuration for MiniMax-M2.1 models.

    Contains parameters specific to the MiniMax-M2 architecture including
    MoE configuration, plus MAX-specific runtime settings.
    """

    # Core model dimensions
    vocab_size: int
    """Vocabulary size of the model."""

    hidden_size: int
    """Dimension of the hidden representations."""

    intermediate_size: int
    """Dimension of the MLP representations per expert."""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer decoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer."""

    num_key_value_heads: int
    """Number of key_value heads for Grouped Query Attention."""

    head_dim: int
    """The attention head dimension."""

    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""

    # Normalization
    rms_norm_eps: float
    """The epsilon used by the RMS normalization layers."""

    # RoPE configuration
    rope_theta: float
    """The base period of the RoPE embeddings."""

    # MoE configuration
    num_local_experts: int
    """Number of local experts in the MoE layer."""

    num_experts_per_tok: int
    """Number of experts to route each token to."""

    # Activation
    hidden_act: str = "silu"
    """The non-linear activation function in the decoder."""

    # Attention configuration
    attention_bias: bool = False
    """Whether to use bias in attention projections."""

    mlp_bias: bool = False
    """Whether to use bias in MLP projections."""

    # MAX-specific parameters
    dtype: DType
    """Data type for model weights and activations."""

    devices: list[DeviceRef]
    """Devices to place the model on."""

    kv_params: KVCacheParams
    """KV cache parameters."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Which logits to return from the model."""

    @override
    def get_kv_params(self) -> KVCacheParams:
        """Get KV cache parameters.

        Returns:
            KV cache parameters configured for this model.
        """
        return self.kv_params

    @override
    def get_max_seq_len(self) -> int:
        """Get maximum sequence length.

        Returns:
            Maximum sequence length supported by the model.
        """
        return self.max_position_embeddings

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Construct KV cache parameters from configuration objects.

        Args:
            huggingface_config: HuggingFace model configuration.
            pipeline_config: MAX pipeline configuration.
            devices: List of devices to use.
            kv_cache_config: KV cache configuration.
            cache_dtype: Data type for the KV cache.

        Returns:
            Constructed KV cache parameters.
        """
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=huggingface_config.num_hidden_layers,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            devices=devices,
            max_seq_len=huggingface_config.max_position_embeddings,
            max_batch_size=kv_cache_config.kv_cache_max_batch_size,
            is_continuous_batching=kv_cache_config.is_continuous_batching,
        )

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initialize the configuration from a pipeline config.

        Args:
            pipeline_config: The pipeline configuration.

        Returns:
            Initialized MiniMaxM2Config instance.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        return cls.initialize_from_config(pipeline_config, huggingface_config)

    @classmethod
    def initialize_from_config(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> Self:
        """Initialize configuration from HuggingFace and pipeline configs.

        Args:
            pipeline_config: The pipeline configuration.
            huggingface_config: The HuggingFace model configuration.

        Returns:
            Initialized MiniMaxM2Config instance.
        """
        # Extract device and dtype configuration
        devices = [DeviceRef(d) for d in pipeline_config.model.execution_devices]
        dtype = DType[pipeline_config.dtype]

        # Determine cache dtype
        cache_dtype_str = (
            pipeline_config.model.kv_cache.kv_cache_dtype
            or pipeline_config.dtype
        )
        cache_dtype = DType[cache_dtype_str]

        # Construct KV cache parameters
        kv_params = cls.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            pipeline_config.model.kv_cache,
            cache_dtype,
        )

        return cls(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            max_position_embeddings=huggingface_config.max_position_embeddings,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_theta=huggingface_config.rope_theta,
            num_local_experts=huggingface_config.num_local_experts,
            num_experts_per_tok=huggingface_config.num_experts_per_tok,
            hidden_act=getattr(
                huggingface_config, "hidden_act", "silu"
            ),
            attention_bias=getattr(
                huggingface_config, "attention_bias", False
            ),
            mlp_bias=getattr(huggingface_config, "mlp_bias", False),
            dtype=dtype,
            devices=devices,
            kv_params=kv_params,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Finalize configuration after weight loading.

        This method is called after weights are loaded to set parameters
        that depend on the actual weights in the state dict.

        Args:
            huggingface_config: HuggingFace model configuration.
            state_dict: The loaded model weights.
            return_logits: Which logits to return from the model.
        """
        self.return_logits = return_logits
