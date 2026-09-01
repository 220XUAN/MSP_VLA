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


def test_build_msp_rope_positions_resets_and_normalizes_inside_each_scale_block():
    scales = (1, 2, 4, 8)
    positions = msp_scale_head.build_msp_rope_positions(
        scales,
        batch_size=2,
        normalization_length=scales[-1],
    )

    expected = jnp.asarray(
        [
            [0, 0, 1 / 8, 0, 1 / 8, 2 / 8, 3 / 8, 0, 1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8],
            [0, 0, 1 / 8, 0, 1 / 8, 2 / 8, 3 / 8, 0, 1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8],
        ],
        dtype=jnp.float32,
    )

    assert positions.shape == expected.shape
    assert jnp.allclose(positions, expected)

    starts, ends = msp_scale_head.build_scale_segment_bounds(scales)
    for block_index, scale in enumerate(scales):
        incremental = msp_scale_head.build_msp_rope_positions(
            (scale,),
            batch_size=2,
            normalization_length=scales[-1],
        )
        assert jnp.allclose(incremental, positions[:, int(starts[block_index]) : int(ends[block_index])])


def test_build_incremental_positions_uses_each_samples_valid_prefix_length():
    prefix_mask = jnp.asarray(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    positions = msp_scale_head.build_incremental_positions(prefix_mask, block_start=3, block_length=4)

    expected = jnp.asarray(
        [
            [6, 7, 8, 9],
            [8, 9, 10, 11],
        ],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(positions, expected)


def test_build_scale_ids_uses_absolute_level_during_incremental_inference():
    scales = (1, 2, 4, 8)

    training_ids = msp_scale_head.build_scale_ids(scales)
    assert jnp.array_equal(training_ids, jnp.asarray([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]))

    for block_index, scale in enumerate(scales):
        inference_ids = msp_scale_head.build_scale_ids((scale,), block_index=block_index)
        assert jnp.array_equal(inference_ids, jnp.full((scale,), block_index, dtype=jnp.int32))


def test_teacher_forcing_and_inference_follow_msp_two_step_resize():
    scales = (1, 2, 4, 8)
    finest = jnp.arange(16, dtype=jnp.float32).reshape(1, 8, 2)

    training_inputs, training_targets = msp_scale_head.build_teacher_forced_inputs(finest, scales)
    starts, ends = msp_scale_head.build_scale_segment_bounds(scales)
    assert training_inputs.shape == training_targets.shape == (1, sum(scales), 2)

    for block_index, scale in enumerate(scales):
        start = int(starts[block_index])
        end = int(ends[block_index])
        expected_target = msp_scale_head.resize_sequence(finest, scale)
        assert jnp.allclose(training_targets[:, start:end], expected_target)

        if block_index == 0:
            assert jnp.array_equal(training_inputs[:, start:end], jnp.zeros_like(expected_target))
            continue

        prev_start = int(starts[block_index - 1])
        prev_end = int(ends[block_index - 1])
        prev_target = training_targets[:, prev_start:prev_end]
        expected_input = msp_scale_head.resize_sequence(
            msp_scale_head.resize_sequence(prev_target, scales[-1]), scale
        )
        assert jnp.allclose(training_inputs[:, start:end], expected_input)

        inference_input = msp_scale_head.build_current_scale_inputs(
            [training_targets[:, int(starts[i]) : int(ends[i])] for i in range(block_index)],
            scales,
            batch_size=1,
            latent_dim=2,
            dtype=jnp.float32,
        )
        assert jnp.allclose(inference_input, expected_input)
