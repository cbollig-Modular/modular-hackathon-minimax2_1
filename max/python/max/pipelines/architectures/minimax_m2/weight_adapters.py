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

import hashlib
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def _get_cache_path(state_dict: dict[str, Weights], huggingface_config: PretrainedConfig) -> Path:
    """Get the cache file path for stacked weights.

    Args:
        state_dict: Weights from checkpoint.
        huggingface_config: Model configuration.

    Returns:
        Path to cache file.
    """
    # Create cache key from model config, weight names, AND dtype
    weight_names = sorted(state_dict.keys())
    # Get dtype from first weight to include in cache key
    first_weight = next(iter(state_dict.values()))
    weight_dtype = str(first_weight.data().dtype)

    config_str = f"{huggingface_config.num_hidden_layers}_{huggingface_config.num_local_experts}_{weight_dtype}"
    cache_key_str = f"{config_str}_{','.join(weight_names[:10])}"  # Use first 10 names
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()

    # Cache directory
    cache_dir = Path.home() / ".cache" / "max" / "stacked_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"minimax_m2_stacked_{cache_key}.pkl"


def _load_cached_weights(cache_path: Path) -> dict[str, WeightData] | None:
    """Load cached stacked weights if available.

    Args:
        cache_path: Path to cache file.

    Returns:
        Cached weights or None if not available.
    """
    if not cache_path.exists():
        return None

    try:
        import logging
        logger = logging.getLogger("max.pipelines")
        logger.info(f"Loading cached stacked weights from {cache_path}")

        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)

        logger.info("Successfully loaded cached stacked weights")
        return cached_data
    except Exception as e:
        import logging
        logger = logging.getLogger("max.pipelines")
        logger.warning(f"Failed to load cache: {e}")
        return None


def _save_cached_weights(cache_path: Path, weights: dict[str, WeightData]) -> None:
    """Save stacked weights to cache.

    Args:
        cache_path: Path to cache file.
        weights: Stacked weights to save.
    """
    try:
        import logging
        logger = logging.getLogger("max.pipelines")
        logger.info(f"Saving stacked weights to cache: {cache_path}")

        with open(cache_path, "wb") as f:
            pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Successfully cached stacked weights for future runs")
    except Exception as e:
        import logging
        logger = logging.getLogger("max.pipelines")
        logger.warning(f"Failed to save cache: {e}")


def _needs_expert_stacking(state_dict: dict[str, Weights]) -> bool:
    """Check if the state dict needs expert weight stacking.

    Args:
        state_dict: Weights loaded from checkpoint.

    Returns:
        True if individual expert weights exist and need to be stacked,
        False if weights are already in stacked format.
    """
    # Check for any individual expert weight (HF format)
    for name in state_dict.keys():
        if ".block_sparse_moe.experts." in name and ".w1.weight" in name:
            return True
    # Check if stacked format already exists
    for name in state_dict.keys():
        if "mlp.experts.gate_up_proj" in name:
            return False
    return False


def _stack_expert_weights_for_layer(
    layer_idx: int,
    experts: dict[int, dict[str, WeightData]],
    scales: dict[int, dict[str, WeightData]] | None = None,
) -> dict[str, WeightData]:
    """Stack expert weights for a single layer.

    This processes one layer at a time to minimize memory usage.

    Args:
        layer_idx: The layer index.
        experts: Dictionary mapping expert_idx to weight dict (w1, w2, w3).
        scales: Optional dictionary of quantization scales.

    Returns:
        Dictionary with stacked weights for this layer.
    """
    num_experts = len(experts)
    if num_experts == 0:
        return {}

    layer_weights = {}

    # Get the dtype from first expert's weight for later restoration
    first_expert = experts[0]
    original_dtype = first_expert["w1"].dtype

    # Debug logging
    import logging
    logger = logging.getLogger("max.pipelines")
    logger.info(f"Layer {layer_idx}: Source weight dtype = {original_dtype}")

    # Get shapes from first expert - convert to plain ints to avoid MLIR context issues
    w1_shape = first_expert["w1"].shape  # [moe_dim, hidden_dim]
    moe_dim = int(w1_shape[0])
    hidden_dim = int(w1_shape[1])

    # Pre-allocate output arrays (more efficient than stack+transpose+concat)
    # gate_up_proj: [num_experts, hidden_dim, 2*moe_dim]
    gate_up_proj = np.empty(
        (num_experts, hidden_dim, 2 * moe_dim),
        dtype=np.uint16 if original_dtype == DType.bfloat16 else np.uint8,
    )

    # down_proj: [num_experts, moe_dim, hidden_dim]
    down_proj = np.empty(
        (num_experts, moe_dim, hidden_dim),
        dtype=np.uint16 if original_dtype == DType.bfloat16 else np.uint8,
    )

    # Fill arrays directly (avoids intermediate copies)
    for expert_idx in range(num_experts):
        expert_data = experts[expert_idx]

        # w1 (gate): [moe_dim, hidden_dim] -> transpose to [hidden_dim, moe_dim]
        w1_np = _weight_data_to_numpy(expert_data["w1"])
        gate_up_proj[expert_idx, :, :moe_dim] = w1_np.T

        # w3 (up): [moe_dim, hidden_dim] -> transpose to [hidden_dim, moe_dim]
        w3_np = _weight_data_to_numpy(expert_data["w3"])
        gate_up_proj[expert_idx, :, moe_dim:] = w3_np.T

        # w2 (down): [hidden_dim, moe_dim] -> transpose to [moe_dim, hidden_dim]
        w2_np = _weight_data_to_numpy(expert_data["w2"])
        down_proj[expert_idx] = w2_np.T

    # Store gate_up_proj (already contiguous since we pre-allocated)
    gate_up_name = f"layers.{layer_idx}.mlp.experts.gate_up_proj"
    layer_weights[gate_up_name] = _numpy_to_weight_data(
        gate_up_proj, gate_up_name, original_dtype
    )

    # Store down_proj (already contiguous)
    down_name = f"layers.{layer_idx}.mlp.experts.down_proj"
    layer_weights[down_name] = _numpy_to_weight_data(
        down_proj, down_name, original_dtype
    )

    # Clear intermediate arrays
    del gate_up_proj, down_proj

    # Handle quantization scales if present
    if scales and any("w1" in scales[i] for i in range(num_experts) if i in scales):
        # Note: Skip FP8 scale tensors - MAX handles quantization internally
        # The checkpoint contains separate scale tensors for FP8 quantization,
        # but MAX's quantization system expects only the quantized weights
        # and handles scaling internally during computation
        scale_dtype = next(
            (scales[i]["w1"].dtype for i in range(num_experts) if i in scales and "w1" in scales[i]),
            None
        )
        if scale_dtype:
            logger.debug(f"Layer {layer_idx}: Skipping FP8 expert scale tensors (MAX handles quantization internally)")
            # Scales are not needed - MAX will handle quantization internally

    return layer_weights


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
    import logging

    logger = logging.getLogger("max.pipelines")

    # Check if we need to stack expert weights
    needs_stacking = _needs_expert_stacking(state_dict)

    if not needs_stacking:
        # Weights are already in the correct format, just apply name mapping
        logger.info("Weights already in stacked format, skipping expert stacking")
        new_state_dict: dict[str, WeightData] = {}
        for name, value in state_dict.items():
            max_name = name
            for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()
        return new_state_dict

    # TODO: Caching disabled due to pickle issues with Buffer objects
    # cache_path = _get_cache_path(state_dict, huggingface_config)
    # cached_weights = _load_cached_weights(cache_path)

    logger.info("Converting expert weights from HuggingFace to stacked format")

    new_state_dict: dict[str, WeightData] = {}

    # Pattern to match expert weights (including quantization scales)
    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight(_scale_inv)?"
    )

    # Get number of decoder layers to filter out MTP weights
    num_decoder_layers = huggingface_config.num_hidden_layers

    # Process weights layer by layer to minimize memory usage
    # First, collect all weight names and categorize them
    expert_weight_names: dict[int, list[str]] = defaultdict(list)
    other_weight_names: list[str] = []

    for safetensor_name in state_dict.keys():
        # Skip MTP (Multi-Token Prediction) module weights
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
            layer_idx = int(match.group(1))
            expert_weight_names[layer_idx].append(safetensor_name)
        else:
            other_weight_names.append(safetensor_name)

    # Process non-expert weights first (these don't need stacking)
    for safetensor_name in other_weight_names:
        # Skip FP8 scale tensors - MAX handles quantization internally
        # These include weight_scale_inv for attention and norm layers
        if "_scale_inv" in safetensor_name or safetensor_name.endswith("_scale"):
            logger.debug(f"Skipping FP8 scale tensor: {safetensor_name}")
            continue

        max_name = safetensor_name
        for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = state_dict[safetensor_name].data()

    # Process expert weights in parallel for speed
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid daemon process issues
    # Numpy releases GIL during heavy operations so threading is still effective
    import os
    max_workers = min(16, (os.cpu_count() or 1) * 2)  # More threads since numpy releases GIL
    logger.info(f"Stacking expert weights using {max_workers} parallel threads")

    # Prepare all layer data first
    all_layer_data = []
    for layer_idx in sorted(expert_weight_names.keys()):
        # Collect weights for this layer only
        layer_expert_weights: dict[int, dict[str, WeightData]] = defaultdict(dict)
        layer_expert_scales: dict[int, dict[str, WeightData]] = defaultdict(dict)

        for safetensor_name in expert_weight_names[layer_idx]:
            match = expert_pattern.match(safetensor_name)
            if not match:
                continue

            expert_idx = int(match.group(2))
            weight_type = match.group(3)  # w1, w2, or w3
            scale_suffix = match.group(4)  # _scale_inv or None

            weight_data = state_dict[safetensor_name].data()

            if scale_suffix:
                layer_expert_scales[expert_idx][weight_type] = weight_data
            else:
                layer_expert_weights[expert_idx][weight_type] = weight_data

        all_layer_data.append((
            layer_idx,
            layer_expert_weights,
            layer_expert_scales if layer_expert_scales else None,
        ))

    # Process layers in parallel with threads (numpy releases GIL for heavy ops)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_layer = {
            executor.submit(
                _stack_expert_weights_for_layer,
                layer_idx,
                experts,
                scales,
            ): layer_idx
            for layer_idx, experts, scales in all_layer_data
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_layer):
            layer_idx = future_to_layer[future]
            stacked_weights = future.result()
            new_state_dict.update(stacked_weights)
            completed += 1
            if completed % 5 == 0 or completed == len(all_layer_data):
                logger.info(
                    f"Stacked {completed}/{num_decoder_layers} layers "
                    f"({100*completed//num_decoder_layers}%)"
                )

    logger.info("Expert weight stacking complete")

    # TODO: Caching disabled due to pickle issues with Buffer objects
    # expert_weights_only = {
    #     k: v for k, v in new_state_dict.items()
    #     if "mlp.experts." in k
    # }
    # _save_cached_weights(cache_path, expert_weights_only)

    return new_state_dict
