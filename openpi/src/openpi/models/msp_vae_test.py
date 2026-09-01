import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import msp_vae


def _small_vae(*, causal: bool = True) -> msp_vae.ActionVAELinen:
    return msp_vae.ActionVAELinen(
        action_dim=6,
        encoder_dim=16,
        decoder_dim=16,
        skill_block_size=8,
        downsample_factor=4,
        attn_pdrop=0.1,
        use_causal_encoder=causal,
        use_causal_decoder=True,
        encoder_heads=2,
        encoder_layers=2,
        decoder_heads=2,
        decoder_layers=2,
        latent_dim=4,
    )


def test_sinusoidal_positions_use_msp_interleaved_layout():
    positions = msp_vae._sinusoidal_positions(2, 6, jnp.float32)
    inv_freq = 1.0 / (10000 ** (np.arange(0, 6, 2, dtype=np.float32) / 6))
    expected = np.stack(
        [
            np.ones(6, dtype=np.float32) * np.asarray([0, 1, 0, 1, 0, 1]),
            np.stack([np.sin(inv_freq), np.cos(inv_freq)], axis=-1).reshape(-1),
        ]
    )

    np.testing.assert_allclose(positions, expected, rtol=1e-6, atol=1e-6)


def test_action_vae_shapes_match_msp_downsampling_and_decode():
    model = _small_vae()
    actions = jnp.ones((2, 8, 6), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        actions,
        jax.random.key(2),
        train=False,
    )

    recon, mean, logvar = model.apply(variables, actions, jax.random.key(3), train=False)
    latent = model.apply(variables, actions, jax.random.key(4), train=False, method=model.get_sample)
    decoded = model.apply(variables, latent, train=False, method=model.get_action)

    assert recon.shape == actions.shape
    assert mean.shape == (2, 2, 4)
    assert logvar.shape == (2, 2, 4)
    assert latent.shape == (2, 2, 4)
    assert decoded.shape == actions.shape


def test_train_mode_dropout_is_active_but_eval_is_deterministic():
    model = _small_vae()
    actions = jnp.ones((2, 8, 6), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        actions,
        jax.random.key(2),
        train=True,
    )

    eval_a = model.apply(variables, actions, jax.random.key(3), train=False)[0]
    eval_b = model.apply(variables, actions, jax.random.key(3), train=False)[0]
    train_a = model.apply(
        variables,
        actions,
        jax.random.key(3),
        train=True,
        rngs={"dropout": jax.random.key(4)},
    )[0]
    train_b = model.apply(
        variables,
        actions,
        jax.random.key(3),
        train=True,
        rngs={"dropout": jax.random.key(5)},
    )[0]

    np.testing.assert_array_equal(eval_a, eval_b)
    assert not np.array_equal(np.asarray(train_a), np.asarray(train_b))


def test_action_projection_uses_pytorch_default_uniform_bounds():
    model = _small_vae()
    actions = jnp.ones((2, 8, 6), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        actions,
        jax.random.key(2),
        train=False,
    )
    kernel = variables["params"]["action_proj"]["kernel"]
    bias = variables["params"]["action_proj"]["bias"]
    bound = 1.0 / np.sqrt(6)

    assert np.max(np.abs(np.asarray(kernel))) <= bound
    assert np.max(np.abs(np.asarray(bias))) <= bound
    assert not np.allclose(bias, 0.0)


def test_noncausal_even_kernel_matches_pytorch_symmetric_padding_length():
    block = msp_vae.Conv1DBlock(features=8, kernel_size=2, stride=1, causal=False)
    x = jnp.ones((1, 5, 8), dtype=jnp.float32)
    variables = block.init(jax.random.key(0), x)
    output = block.apply(variables, x)

    # PyTorch Conv1d(padding=kernel_size // 2) produces L + 1 for an even kernel.
    assert output.shape == (1, 6, 8)
