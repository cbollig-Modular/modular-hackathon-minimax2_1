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

"""Weight adapters for converting HuggingFace MiniMax-M2 weights to MAX format."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

# Maps from HuggingFace Safetensor to MAX weight names.
MINIMAX_M2_SAFETENSOR_MAP = {
    "model.": "",  # Removes the "model" prefix
    ".block_sparse_moe.gate.weight": ".mlp.gate.gate_score.weight",  # MoE gate weight
    ".block_sparse_moe.e_score_correction_bias": ".mlp.gate.e_score_correction_bias",  # MoE bias
}


def _weight_data_to_numpy(weight_data: WeightData) -> np.ndarray:
    """Convert WeightData to numpy array, handling special dtypes.

    numpy's from_dlpack doesn't support bfloat16 and some other dtypes,
    so we use MAX's Buffer to handle the conversion.

    Args:
        weight_data: The weight data to convert.

    Returns:
        Numpy array with the weight data.
    """
    buffer = Buffer.from_dlpack(weight_data.data)

    # For dtypes not supported by numpy, we need to use a compatible view
    if weight_data.dtype == DType.bfloat16:
        # View as uint16 (same 16-bit layout) for stacking operations
        buffer = buffer.view(dtype=DType.uint16, shape=buffer.shape)
    elif weight_data.dtype.is_float8():
        # Float8 types need to be viewed as uint8
        buffer = buffer.view(dtype=DType.uint8, shape=buffer.shape)

    return buffer.to_numpy()


def _numpy_to_weight_data(
    arr: np.ndarray, name: str, original_dtype: DType
) -> WeightData:
    """Convert numpy array back to WeightData, handling special dtypes.

    Args:
        arr: Numpy array (may be uint16/uint8 view for special dtypes).
        name: Name for the weight.
        original_dtype: Original dtype to restore.

    Returns:
        WeightData with the correct dtype.
    """
    buffer = Buffer.from_numpy(arr)

    # View back to original dtype if needed
    if original_dtype == DType.bfloat16:
        # The array is uint16 view of bfloat16 data
        buffer = buffer.view(dtype=DType.bfloat16, shape=buffer.shape)
    elif original_dtype.is_float8():
        # The array is uint8 view of float8 data
        buffer = buffer.view(dtype=original_dtype, shape=buffer.shape)

    return WeightData(
        data=buffer,
        name=name,
        dtype=original_dtype,
        shape=Shape(buffer.shape),
    )


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert HuggingFace MiniMax-M2 weights to MAX format.

    This function maps HuggingFace weight keys to MAX internal format and
    handles MoE-specific weight transformations. It also removes MTP
    (Multi-Token Prediction) module weights which are not used in inference.

    HuggingFace MiniMax-M2 format:
        - model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight: [moe_dim, hidden_dim] (gate)
        - model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight: [hidden_dim, moe_dim] (down)
        - model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight: [moe_dim, hidden_dim] (up)
        - model.layers.{i}.block_sparse_moe.gate.weight: [num_experts, hidden_dim]

    MAX MiniMaxM2 expected format:
        - layers.{i}.mlp.experts.gate_up_proj: [num_experts, hidden_dim, 2*moe_dim]
        - layers.{i}.mlp.experts.down_proj: [num_experts, moe_dim, hidden_dim]
        - layers.{i}.mlp.gate.gate_score.weight: [num_experts, hidden_dim]

    Args:
        state_dict: Weights loaded from HuggingFace checkpoint.
        huggingface_config: Model configuration from HuggingFace.
        **unused_kwargs: Additional unused keyword arguments.

    Returns:
        Dictionary mapping MAX weight names to WeightData objects.
    """
    new_state_dict: dict[str, WeightData] = {}

    # Pattern to match expert weights (including quantization scales)
    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight(_scale_inv)?"
    )

    # Collect expert weights by layer for stacking
    # Structure: {layer_idx: {expert_idx: {weight_type: weight_data}}}
    expert_weights: dict[int, dict[int, dict[str, WeightData]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    expert_scales: dict[int, dict[int, dict[str, WeightData]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Get number of decoder layers to filter out MTP weights
    num_decoder_layers = huggingface_config.num_hidden_layers

    # First pass: separate expert weights from other weights
    for safetensor_name, value in state_dict.items():
        # Skip MTP (Multi-Token Prediction) module weights
        # MTP modules appear after the main decoder layers
        if safetensor_name.startswith("model.layers."):
            parts = safetensor_name.split(".")
            if len(parts) >= 3:
                try:
                    layer_idx = int(parts[2])
                    if layer_idx >= num_decoder_layers:
                        continue  # Skip MTP weights
                except ValueError:
                    pass

        match = expert_pattern.match(safetensor_name)

        if match:
            # This is an expert weight - collect it for stacking
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            weight_type = match.group(3)  # w1, w2, or w3
            scale_suffix = match.group(4)  # _scale_inv or None

            if scale_suffix:
                # This is a quantization scale
                expert_scales[layer_idx][expert_idx][weight_type] = value.data()
            else:
                # This is a regular weight
                expert_weights[layer_idx][expert_idx][weight_type] = value.data()
        else:
            # Standard weight - apply name mapping
            max_name = safetensor_name
            for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()

    # Second pass: stack expert weights
    for layer_idx in sorted(expert_weights.keys()):
        experts = expert_weights[layer_idx]
        num_experts = len(experts)

        if num_experts == 0:
            continue

        # Collect and stack gate (w1), up (w3), and down (w2) projections
        w1_projs = []  # gate projections
        w2_projs = []  # down projections
        w3_projs = []  # up projections

        # Get the dtype from first expert's weight for later restoration
        first_expert = experts[0]
        original_dtype = first_expert["w1"].dtype

        for expert_idx in range(num_experts):
            expert_data = experts[expert_idx]
            w1_projs.append(_weight_data_to_numpy(expert_data["w1"]))
            w2_projs.append(_weight_data_to_numpy(expert_data["w2"]))
            w3_projs.append(_weight_data_to_numpy(expert_data["w3"]))

        # Stack gate (w1) and up (w3) projections
        # HF format: [moe_dim, hidden_dim] per expert
        # Target format: [num_experts, hidden_dim, 2*moe_dim]

        # Stack all experts: [num_experts, moe_dim, hidden_dim]
        stacked_w1 = np.stack(w1_projs, axis=0)
        stacked_w3 = np.stack(w3_projs, axis=0)

        # Transpose to [num_experts, hidden_dim, moe_dim]
        stacked_w1 = np.transpose(stacked_w1, (0, 2, 1))
        stacked_w3 = np.transpose(stacked_w3, (0, 2, 1))

        # Concatenate w1 (gate) and w3 (up) along last dimension
        # Result: [num_experts, hidden_dim, 2*moe_dim]
        gate_up_proj = np.concatenate([stacked_w1, stacked_w3], axis=2)
        gate_up_proj = np.ascontiguousarray(gate_up_proj)

        # Store gate_up_proj
        gate_up_name = f"layers.{layer_idx}.mlp.experts.gate_up_proj"
        new_state_dict[gate_up_name] = _numpy_to_weight_data(
            gate_up_proj, gate_up_name, original_dtype
        )

        # Stack down (w2) projections
        # HF format: [hidden_dim, moe_dim] per expert
        # Target format: [num_experts, moe_dim, hidden_dim]
        stacked_w2 = np.stack(w2_projs, axis=0)
        # Transpose to [num_experts, moe_dim, hidden_dim]
        stacked_w2 = np.transpose(stacked_w2, (0, 2, 1))
        stacked_w2 = np.ascontiguousarray(stacked_w2)

        down_name = f"layers.{layer_idx}.mlp.experts.down_proj"
        new_state_dict[down_name] = _numpy_to_weight_data(
            stacked_w2, down_name, original_dtype
        )

        # Handle quantization scales if present
        if layer_idx in expert_scales:
            scales = expert_scales[layer_idx]
            if scales:
                # Stack quantization scales similarly
                w1_scales = []
                w2_scales = []
                w3_scales = []

                for expert_idx in range(num_experts):
                    if expert_idx in scales:
                        scale_data = scales[expert_idx]
                        if "w1" in scale_data:
                            w1_scales.append(_weight_data_to_numpy(scale_data["w1"]))
                        if "w2" in scale_data:
                            w2_scales.append(_weight_data_to_numpy(scale_data["w2"]))
                        if "w3" in scale_data:
                            w3_scales.append(_weight_data_to_numpy(scale_data["w3"]))

                if w1_scales and w3_scales:
                    # Stack and combine gate_up scales
                    stacked_w1_scale = np.stack(w1_scales, axis=0)
                    stacked_w3_scale = np.stack(w3_scales, axis=0)
                    stacked_w1_scale = np.transpose(stacked_w1_scale, (0, 2, 1))
                    stacked_w3_scale = np.transpose(stacked_w3_scale, (0, 2, 1))
                    gate_up_scale = np.concatenate([stacked_w1_scale, stacked_w3_scale], axis=2)
                    gate_up_scale = np.ascontiguousarray(gate_up_scale)

                    gate_up_scale_name = f"layers.{layer_idx}.mlp.experts.gate_up_proj_scale"
                    scale_dtype = scales[0]["w1"].dtype
                    new_state_dict[gate_up_scale_name] = _numpy_to_weight_data(
                        gate_up_scale, gate_up_scale_name, scale_dtype
                    )

                if w2_scales:
                    # Stack down scales
                    stacked_w2_scale = np.stack(w2_scales, axis=0)
                    stacked_w2_scale = np.transpose(stacked_w2_scale, (0, 2, 1))
                    stacked_w2_scale = np.ascontiguousarray(stacked_w2_scale)

                    down_scale_name = f"layers.{layer_idx}.mlp.experts.down_proj_scale"
                    scale_dtype = scales[0]["w2"].dtype
                    new_state_dict[down_scale_name] = _numpy_to_weight_data(
                        stacked_w2_scale, down_scale_name, scale_dtype
                    )

    return new_state_dict
