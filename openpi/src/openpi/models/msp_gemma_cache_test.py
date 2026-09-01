import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import msp_scale_head


def _make_tiny_msp_gemma():
    config = gemma.Config(
        width=8,
        depth=2,
        mlp_dim=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    llm = nnx_bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="float32"))
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])
    return llm


def test_full_suffix_matches_scale_wise_kv_cache_forward():
    scales = (1, 2, 4)
    batch_size = 2
    prefix_len = 3
    width = 8
    total_suffix_len = sum(scales)
    llm = _make_tiny_msp_gemma()

    prefix_tokens = jax.random.normal(jax.random.key(1), (batch_size, prefix_len, width))
    suffix_tokens = jax.random.normal(jax.random.key(2), (batch_size, total_suffix_len, width))
    prefix_mask = jnp.asarray(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    suffix_mask = jnp.ones((batch_size, total_suffix_len), dtype=jnp.bool_)
    full_suffix_mask = msp_scale_head.build_suffix_block_attention_mask(
        scales,
        batch_size=batch_size,
        input_mask=suffix_mask,
    )
    full_attention_mask = msp_scale_head.build_full_attention_mask(
        prefix_mask,
        suffix_mask,
        full_suffix_mask,
    )
    full_positions = jnp.cumsum(jnp.concatenate([prefix_mask, suffix_mask], axis=1), axis=1) - 1
    full_rope_positions = msp_scale_head.build_msp_rope_positions(
        scales,
        batch_size=batch_size,
        normalization_length=scales[-1],
    )

    (_, full_suffix_out), _ = llm(
        [prefix_tokens, suffix_tokens],
        mask=full_attention_mask,
        positions=full_positions,
        adarms_cond=[None, None],
        rope_positions=[None, full_rope_positions],
    )

    prefix_attention_mask = prefix_mask[:, :, None] & prefix_mask[:, None, :]
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = llm(
        [prefix_tokens, None],
        mask=prefix_attention_mask,
        positions=prefix_positions,
        adarms_cond=[None, None],
    )

    starts, ends = msp_scale_head.build_scale_segment_bounds(scales)
    cached_blocks = []
    for block_index, scale in enumerate(scales):
        start = int(starts[block_index])
        end = int(ends[block_index])
        current_tokens = suffix_tokens[:, start:end]
        current_suffix_mask = msp_scale_head.build_current_block_attention_mask(
            scales,
            block_index,
            batch_size=batch_size,
        )
        current_prefix_mask = jnp.broadcast_to(prefix_mask[:, None, :], (batch_size, scale, prefix_len))
        current_attention_mask = jnp.concatenate([current_prefix_mask, current_suffix_mask], axis=-1)
        current_positions = msp_scale_head.build_incremental_positions(
            prefix_mask,
            block_start=start,
            block_length=scale,
        )
        current_rope_positions = msp_scale_head.build_msp_rope_positions(
            (scale,),
            batch_size=batch_size,
            normalization_length=scales[-1],
        )

        (_, current_out), kv_cache = llm(
            [None, current_tokens],
            mask=current_attention_mask,
            positions=current_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, None],
            rope_positions=[None, current_rope_positions],
        )
        cached_blocks.append(current_out)

    cached_suffix_out = jnp.concatenate(cached_blocks, axis=1)
    assert jnp.allclose(cached_suffix_out, full_suffix_out, atol=1e-5, rtol=1e-5)
