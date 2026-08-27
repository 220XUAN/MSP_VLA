import math

import jax
import jax.numpy as jnp


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
    mask = []
    for scale in scales:
        mask.extend([True] + ([False] * (scale - 1)))
    return jnp.asarray(mask, dtype=jnp.bool_)


def build_block_local_positions(scales: tuple[int, ...], *, batch_size: int) -> jnp.ndarray:
    positions = jnp.concatenate([jnp.arange(scale, dtype=jnp.int32) for scale in scales], axis=0)
    return jnp.broadcast_to(positions[None, :], (batch_size, positions.shape[0]))


def build_scale_segment_bounds(scales: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    ends = jnp.cumsum(jnp.asarray(scales, dtype=jnp.int32))
    starts = ends - jnp.asarray(scales, dtype=jnp.int32)
    return starts, ends


def build_suffix_block_attention_mask(
    scales: tuple[int, ...],
    *,
    batch_size: int,
    input_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    total_length = sum(scales)
    rows = []
    visible = 0
    for scale in scales:
        visible += scale
        row = jnp.concatenate(
            [
                jnp.ones((scale, visible), dtype=jnp.bool_),
                jnp.zeros((scale, total_length - visible), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        rows.append(row)
    mask = jnp.concatenate(rows, axis=0)
    mask = jnp.broadcast_to(mask[None, :, :], (batch_size, total_length, total_length))
    if input_mask is not None:
        valid = input_mask[:, :, None] & input_mask[:, None, :]
        mask = mask & valid
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


def build_scale_ids(scales: tuple[int, ...]) -> jnp.ndarray:
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
