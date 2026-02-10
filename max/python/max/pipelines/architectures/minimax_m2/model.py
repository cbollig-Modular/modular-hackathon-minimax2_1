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

"""MiniMax-M2.1 pipeline model implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheInputsSequence, KVCacheParams, PagedCacheValues
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig

from .minimax_m2 import MiniMaxM2
from .model_config import MiniMaxM2Config

logger = logging.getLogger("max.pipelines")


class MiniMaxM2Inputs(ModelInputs):
    """Input tensors for the MiniMax-M2.1 model."""

    tokens: Buffer
    """Input token IDs."""

    input_row_offsets: Buffer
    """Row offsets for ragged batching."""

    return_n_logits: Buffer
    """Number of logits to return."""

    def __init__(
        self,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        """Initialize model inputs.

        Args:
            tokens: Input token IDs.
            input_row_offsets: Row offsets for ragged batching.
            return_n_logits: Number of logits to return.
            kv_cache_inputs: KV cache inputs.
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs


class MiniMaxM2Model(PipelineModel[TextContext], KVCacheMixin):
    """MiniMax-M2.1 pipeline model for text generation.

    Integrates the MiniMax-M2.1 architecture with MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and
    inference execution.
    """

    model: Model
    """The compiled MAX Engine model ready for inference."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        """Initialize the MiniMax-M2.1 pipeline model.

        Args:
            pipeline_config: Pipeline configuration settings.
            session: MAX Engine inference session.
            huggingface_config: HuggingFace model configuration.
            encoding: Quantization and data type encoding.
            devices: List of devices to run the model on.
            kv_cache_config: KV cache configuration.
            weights: Model weights.
            adapter: Optional weight adapter.
            return_logits: Which logits to return.
        """
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace configuration.

        Returns:
            Maximum sequence length.
        """
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters.

        Args:
            huggingface_config: HuggingFace configuration.
            pipeline_config: Pipeline configuration.
            devices: List of devices.
            kv_cache_config: KV cache configuration.
            cache_dtype: Cache data type.

        Returns:
            KV cache parameters.
        """
        return MiniMaxM2Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Get number of hidden layers.

        Args:
            huggingface_config: HuggingFace configuration.

        Returns:
            Number of hidden layers.
        """
        return huggingface_config.num_hidden_layers

    def load_model(self, session: InferenceSession) -> Model:
        """Load the compiled model.

        Args:
            session: MAX Engine inference session.

        Returns:
            Loaded MAX Engine model.
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        timer = CompilationTimer("model")
        graph = self._build_graph()
        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

        return model

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: list[Any]
    ) -> PagedCacheValues:
        """Unflatten KV cache inputs.

        Args:
            kv_inputs_flat: Flattened KV inputs.

        Returns:
            PagedCacheValues object.
        """
        return PagedCacheValues(
            kv_blocks=kv_inputs_flat[0].buffer,
            cache_lengths=kv_inputs_flat[1].tensor,
            lookup_table=kv_inputs_flat[2].tensor,
            max_lengths=kv_inputs_flat[3].tensor,
        )

    def _build_graph(self) -> Graph:
        """Build the computation graph.

        Returns:
            Compiled computation graph.
        """
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        # Define input types
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Load and adapt weights
        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        # Initialize model configuration
        model_config = MiniMaxM2Config.initialize_from_config(
            self.pipeline_config, huggingface_config
        )
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )

        # Create model and load weights
        nn_model = MiniMaxM2(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        self.state_dict = nn_model.state_dict(auto_initialize=False)

        # Get KV cache input types
        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        # Build computation graph
        with Graph(
            "MiniMaxM2ForCausalLM",
            input_types=[
                tokens_type,
                return_n_logits_type,
                input_row_offsets_type,
                *flattened_kv_types,
            ],
        ) as graph:
            # Unpack inputs
            tokens, return_n_logits, input_row_offsets, *kv_inputs_flat = (
                graph.inputs
            )

            # Unflatten KV cache inputs
            kv_cache = self._unflatten_kv_inputs(kv_inputs_flat)

            # Execute model
            logits = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=input_row_offsets.tensor,
                kv_collection=kv_cache,
            )

            graph.output(logits)

        return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model with prepared inputs.

        Args:
            model_inputs: Prepared model inputs.

        Returns:
            Model outputs containing logits.
        """
        model_inputs = cast(MiniMaxM2Inputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        # Handle input_row_offsets conversion if needed
        if isinstance(model_inputs.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(model_inputs.input_row_offsets).to(
                self.devices[0]
            )
        else:
            input_row_offsets = model_inputs.input_row_offsets

        # Execute the model
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            input_row_offsets,
            *curr_kv_cache_inputs,
        )

        # Extract logits from outputs
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1]),
                next_token_logits=cast(Buffer, model_outputs[0]),
                logit_offsets=cast(Buffer, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0]),
                next_token_logits=cast(Buffer, model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepare inputs for the first execution pass.

        Args:
            replica_batches: Sequence of TextContext batches for each replica.
            kv_cache_inputs: Optional KV cache inputs.
            return_n_logits: Number of logits to return.

        Returns:
            Prepared ModelInputs for initial execution.
        """
        if len(replica_batches) > 1:
            raise ValueError("Model does not support data parallelism > 1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)

        # Get input_row_offsets: start and end position of each batch
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create ragged token vector
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        # Create input_row_offsets tensor
        input_row_offsets_tensor = Buffer.from_numpy(input_row_offsets).to(
            self.devices[0]
        )

        return MiniMaxM2Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=input_row_offsets_tensor,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        """Prepare inputs for subsequent execution steps.

        Args:
            next_tokens: Token IDs generated in the previous step.
            prev_model_inputs: ModelInputs from previous step.

        Returns:
            Prepared ModelInputs for next execution step.
        """
        prev_model_inputs = cast(MiniMaxM2Inputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]

        next_row_offsets = self._input_row_offsets_prealloc[
            :row_offsets_size
        ].to(self.devices[0])

        return MiniMaxM2Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
