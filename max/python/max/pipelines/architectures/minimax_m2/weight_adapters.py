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

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

# Maps from HuggingFace Safetensor to MAX weight names.
MINIMAX_M2_SAFETENSOR_MAP = {
    "model.": "",  # Removes the "model" prefix
    ".mlp.gate.weight": ".mlp.gate.gate_score.weight",  # MoE gate weight
    ".mlp.e_score_correction_bias": ".mlp.gate.e_score_correction_bias",  # MoE bias
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert HuggingFace MiniMax-M2 weights to MAX format.

    This function maps HuggingFace weight keys to MAX internal format and
    handles MoE-specific weight transformations. It also removes MTP
    (Multi-Token Prediction) module weights which are not used in inference.

    Args:
        state_dict: Weights loaded from HuggingFace checkpoint.
        huggingface_config: Model configuration from HuggingFace.
        **unused_kwargs: Additional unused keyword arguments.

    Returns:
        Dictionary mapping MAX weight names to WeightData objects.
    """
    new_state_dict: dict[str, WeightData] = {}

    # Map the weight names from HuggingFace to MAX format
    for name, value in state_dict.items():
        max_name = name
        for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    # Remove MTP (Multi-Token Prediction) module weights
    # MTP modules are additional prediction heads used during training
    # but are not needed for inference (only the main LM head is used)
    # In MiniMax-M2, MTP modules appear after the main decoder layers
    num_decoder_layers = huggingface_config.num_hidden_layers

    # MTP modules in MiniMax-M2 are stored as layers beyond num_hidden_layers
    # e.g., if there are 62 decoder layers, MTP would be layers 62, 63, 64
    # We need to remove all weights for these MTP layers
    keys_to_remove = []
    for key in new_state_dict.keys():
        # Check if this key belongs to an MTP layer
        if key.startswith("layers."):
            # Extract layer index from key like "layers.62.something"
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    if layer_idx >= num_decoder_layers:
                        keys_to_remove.append(key)
                except ValueError:
                    # Not a numeric layer index, skip
                    pass

    # Remove MTP weights
    for key in keys_to_remove:
        del new_state_dict[key]

    return new_state_dict
