import jax.numpy as jnp

from openpi.models import msp_scale_head


def test_build_blockwise_visibility_matches_msp_layout():
    scales = (1, 2, 4, 8)

    mask = msp_scale_head.build_blockwise_visibility(scales)

    expected = jnp.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=jnp.bool_,
    )

    assert mask.shape == expected.shape
    assert jnp.array_equal(mask, expected)


def test_build_current_block_attention_mask_matches_training_rows():
    scales = (1, 2, 4, 8)
    full_mask = msp_scale_head.build_blockwise_visibility(scales)
    starts, ends = msp_scale_head.build_scale_segment_bounds(scales)

    for block_index in range(len(scales)):
        start = int(starts[block_index])
        end = int(ends[block_index])
        block_mask = msp_scale_head.build_current_block_attention_mask(scales, block_index, batch_size=2)

        assert block_mask.shape == (2, end - start, end)
        expected = jnp.broadcast_to(full_mask[start:end, :end][None, :, :], block_mask.shape)
        assert jnp.array_equal(block_mask, expected)


def test_slice_scale_positions_uses_full_table_for_training_and_block_slice_for_inference():
    scales = (1, 2, 4, 8)
    pos = jnp.arange(sum(scales) * 3, dtype=jnp.float32).reshape(1, sum(scales), 3)

    full = msp_scale_head.slice_scale_positions(pos, scales, block_index=None)
    assert jnp.array_equal(full, pos)

    starts, ends = msp_scale_head.build_scale_segment_bounds(scales)
    for block_index in range(len(scales)):
        start = int(starts[block_index])
        end = int(ends[block_index])
        sliced = msp_scale_head.slice_scale_positions(pos, scales, block_index=block_index)
        assert jnp.array_equal(sliced, pos[:, start:end, :])


def test_build_block_local_positions_resets_inside_each_scale_block():
    scales = (1, 2, 4, 8)
    positions = msp_scale_head.build_block_local_positions(scales, batch_size=2)

    expected = jnp.asarray(
        [
            [0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=jnp.int32,
    )

    assert positions.shape == expected.shape
    assert jnp.array_equal(positions, expected)
