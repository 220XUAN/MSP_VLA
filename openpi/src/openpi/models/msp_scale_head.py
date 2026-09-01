import math

import jax
import jax.numpy as jnp
import numpy as np


def default_msp_scales(latent_horizon: int) -> tuple[int, ...]:
    if latent_horizon < 1:
        raise ValueError("latent_horizon must be positive")
    scales = []
    scale = 1
    while scale < latent_horizon:
        scales.append(scale)
        scale *= 2
    if not scales or scales[-1] != latent_horizon:
        scales.append(latent_horizon)
    return tuple(scales)


def resize_sequence(x: jnp.ndarray, target_length: int) -> jnp.ndarray:
    if x.shape[1] == target_length:
        return x
    return jax.image.resize(x, (x.shape[0], target_length, x.shape[2]), method="linear")


def build_scale_ar_mask(scales: tuple[int, ...]) -> jnp.ndarray:
    """Compatibility helper for big_vision-style blockwise AR masks.

    `True` marks the first token of a new scale block. This makes cumsum-based
    attention treat each scale as one fully-connected block that can attend to
    all previous blocks.
    """
    mask = []
    for scale in scales:
        mask.extend([True] + ([False] * (scale - 1)))
    return jnp.asarray(mask, dtype=jnp.bool_)


def build_msp_rope_positions(
    scales: tuple[int, ...], *, batch_size: int, normalization_length: int
) -> jnp.ndarray:
    """Build MSP block-local RoPE positions normalized by the finest scale."""
    if normalization_length < max(scales):
        raise ValueError(
            f"normalization_length must cover every scale, got {normalization_length} for {scales}"
        )
    positions = jnp.concatenate([jnp.arange(scale, dtype=jnp.float32) for scale in scales], axis=0)
    positions = positions / float(normalization_length)
    return jnp.broadcast_to(positions[None, :], (batch_size, positions.shape[0]))


def build_incremental_positions(
    prefix_mask: jnp.ndarray, *, block_start: int, block_length: int
) -> jnp.ndarray:
    """Build global cache-layout positions from each sample's valid prefix length."""
    prefix_offsets = jnp.sum(prefix_mask, axis=-1, dtype=jnp.int32)[:, None]
    block_positions = block_start + jnp.arange(block_length, dtype=jnp.int32)[None, :]
    return prefix_offsets + block_positions


def build_scale_segment_bounds(scales: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Returns static numpy arrays so segment bounds stay concrete under jit."""
    ends = np.cumsum(scales, dtype=np.int32)
    starts = ends - np.asarray(scales, dtype=np.int32)
    return starts, ends


def build_blockwise_visibility(scales: tuple[int, ...]) -> jnp.ndarray:
    """Build the exact MSP block-wise suffix visibility matrix.

    For each scale block, all tokens in that block can attend to:
    - every token in coarser scales
    - every token in the same scale
    - no token in finer scales
    """
    total_length = sum(scales)
    rows = []
    visible = 0
    for scale in scales:
        visible += scale
        rows.append(
            jnp.concatenate(
                [
                    jnp.ones((scale, visible), dtype=jnp.bool_),
                    jnp.zeros((scale, total_length - visible), dtype=jnp.bool_),
                ],
                axis=-1,
            )
        )
    return jnp.concatenate(rows, axis=0)


def build_suffix_block_attention_mask(
    scales: tuple[int, ...],
    *,
    batch_size: int,
    input_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    total_length = sum(scales)
    mask = build_blockwise_visibility(scales)
    mask = jnp.broadcast_to(mask[None, :, :], (batch_size, total_length, total_length))
    if input_mask is not None:
        valid = input_mask[:, :, None] & input_mask[:, None, :]
        mask = mask & valid
    return mask


def build_current_block_attention_mask(
    scales: tuple[int, ...],
    block_index: int,
    *,
    batch_size: int,
    input_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Build inference-time suffix attention mask for the current scale block."""
    visibility = build_blockwise_visibility(scales)
    starts, ends = build_scale_segment_bounds(scales)
    start = int(starts[block_index])
    end = int(ends[block_index])
    mask = visibility[start:end, :end]
    mask = jnp.broadcast_to(mask[None, :, :], (batch_size, end - start, end))
    if input_mask is not None:
        valid_query = input_mask[:, start:end, None]
        valid_key = input_mask[:, None, :end]
        mask = mask & valid_query & valid_key
    return mask


def build_full_attention_mask(
    prefix_mask: jnp.ndarray,
    suffix_mask: jnp.ndarray,
    suffix_attention_mask: jnp.ndarray,
) -> jnp.ndarray:
    batch_size, prefix_len = prefix_mask.shape
    suffix_len = suffix_mask.shape[1]
    prefix_prefix = prefix_mask[:, :, None] & prefix_mask[:, None, :]
    prefix_suffix = jnp.zeros((batch_size, prefix_len, suffix_len), dtype=jnp.bool_)
    suffix_prefix = jnp.broadcast_to(prefix_mask[:, None, :], (batch_size, suffix_len, prefix_len))
    suffix_suffix = suffix_attention_mask & (suffix_mask[:, :, None] & suffix_mask[:, None, :])
    top = jnp.concatenate([prefix_prefix, prefix_suffix], axis=-1)
    bottom = jnp.concatenate([suffix_prefix, suffix_suffix], axis=-1)
    return jnp.concatenate([top, bottom], axis=1)


def build_scale_ids(scales: tuple[int, ...], *, block_index: int | None = None) -> jnp.ndarray:
    """Build level IDs for full-sequence training or one inference block."""
    if block_index is not None:
        if len(scales) != 1:
            raise ValueError("block_index requires exactly one current inference scale")
        return jnp.full((scales[0],), block_index, dtype=jnp.int32)
    return jnp.concatenate([jnp.full((scale,), i, dtype=jnp.int32) for i, scale in enumerate(scales)], axis=0)


def build_temporal_positions(scales: tuple[int, ...]) -> jnp.ndarray:
    return jnp.concatenate([jnp.arange(scale, dtype=jnp.int32) for scale in scales], axis=0)


def build_multiscale_targets(finest_latent: jnp.ndarray, scales: tuple[int, ...]) -> list[jnp.ndarray]:
    return [resize_sequence(finest_latent, scale) for scale in scales]


def build_teacher_forced_inputs(finest_latent: jnp.ndarray, scales: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    targets = build_multiscale_targets(finest_latent, scales)
    inputs = []
    finest_scale = scales[-1]
    for i, target in enumerate(targets):
        if i == 0:
            inputs.append(jnp.zeros_like(target))
        else:
            prev = targets[i - 1]
            prev = resize_sequence(resize_sequence(prev, finest_scale), target.shape[1])
            inputs.append(prev)
    return jnp.concatenate(inputs, axis=1), jnp.concatenate(targets, axis=1)


def build_partial_inputs(generated_blocks: list[jnp.ndarray], scales: tuple[int, ...]) -> jnp.ndarray:
    inputs = []
    finest_scale = scales[-1]
    for i in range(len(generated_blocks) + 1):
        scale = scales[i]
        if i == 0:
            inputs.append(jnp.zeros((generated_blocks[0].shape[0], scale, generated_blocks[0].shape[2]), dtype=generated_blocks[0].dtype))
        else:
            prev = generated_blocks[i - 1]
            prev = resize_sequence(resize_sequence(prev, finest_scale), scale)
            inputs.append(prev)
    return jnp.concatenate(inputs, axis=1)


def build_current_scale_inputs(
    generated_blocks: list[jnp.ndarray],
    scales: tuple[int, ...],
    *,
    batch_size: int,
    latent_dim: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    current_scale = scales[len(generated_blocks)]
    if not generated_blocks:
        return jnp.zeros((batch_size, current_scale, latent_dim), dtype=dtype)
    prev = generated_blocks[-1]
    prev = resize_sequence(resize_sequence(prev, scales[-1]), current_scale)
    return prev


def latent_horizon_from_action_horizon(action_horizon: int, downsample_factor: int) -> int:
    return math.ceil(action_horizon / downsample_factor)
