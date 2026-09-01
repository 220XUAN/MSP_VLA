import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import msp_flow_head


def test_sample_time_pairs_matches_msp_constraints():
    t, r = msp_flow_head.sample_time_pairs(jax.random.key(0), 10, flow_ratio=0.5)

    assert t.shape == (10,)
    assert r.shape == (10,)
    assert np.all(np.asarray(t) >= np.asarray(r))
    assert np.sum(np.asarray(t) == np.asarray(r)) == 5
    assert np.all((np.asarray(t) > 0.0) & (np.asarray(t) < 1.0))


def test_adaptive_l2_loss_returns_per_example_original_formula():
    error = jnp.asarray([[[1.0, 1.0]], [[2.0, 2.0]]])
    loss = msp_flow_head.adaptive_l2_loss(error, gamma=0.5, c=1e-3)
    delta_sq = np.asarray([1.0, 4.0])
    expected = delta_sq / np.sqrt(delta_sq + 1e-3)

    np.testing.assert_allclose(loss, expected, rtol=1e-6)


def test_flow_model_and_meanflow_loss_have_expected_shapes():
    model = msp_flow_head.MspScaleFlowHead(
        latent_dim=4,
        condition_dim=8,
        hidden_dim=16,
        time_dim=6,
        time_hidden_dim=8,
        num_blocks=2,
        dropout_rate=0.1,
    )
    sample = jnp.ones((3, 2, 4), dtype=jnp.float32)
    condition = jnp.ones((3, 2, 8), dtype=jnp.float32)
    timestep = jnp.ones((3,), dtype=jnp.float32)
    r = jnp.zeros((3,), dtype=jnp.float32)
    params = model.init(jax.random.key(0), sample, timestep, condition, r, jax.random.key(1), train=False)

    def apply_model(sample, timestep, condition, r, dropout_rng, *, train):
        return model.apply(params, sample, timestep, condition, r, dropout_rng, train=train)

    output = apply_model(sample, timestep, condition, r, jax.random.key(2), train=False)
    loss, mse = msp_flow_head.meanflow_loss(
        apply_model,
        sample,
        condition,
        jax.random.key(3),
        train=True,
    )
    jitted_loss, jitted_mse = jax.jit(
        lambda target, cond, rng: msp_flow_head.meanflow_loss(
            apply_model,
            target,
            cond,
            rng,
            train=True,
        )
    )(sample, condition, jax.random.key(4))

    assert output.shape == sample.shape
    assert loss.shape == (3,)
    assert mse.shape == (3,)
    assert np.all(np.isfinite(np.asarray(loss)))
    assert np.all(np.isfinite(np.asarray(mse)))
    assert np.all(np.isfinite(np.asarray(jitted_loss)))
    assert np.all(np.isfinite(np.asarray(jitted_mse)))


def test_generate_matches_original_single_step_update():
    def constant_velocity(sample, timestep, condition, r, dropout_rng, *, train):
        del timestep, condition, r, dropout_rng, train
        return jnp.full_like(sample, 0.25)

    sample = jnp.ones((2, 3, 4), dtype=jnp.float32)
    condition = jnp.zeros((2, 3, 8), dtype=jnp.float32)

    generated = msp_flow_head.generate(constant_velocity, condition, sample)

    np.testing.assert_allclose(generated, sample - 0.25)


def test_nnx_bridge_supports_meanflow_jvp_and_gradients():
    flow_head = nnx_bridge.ToNNX(
        msp_flow_head.MspScaleFlowHead(
            latent_dim=4,
            condition_dim=8,
            hidden_dim=16,
            time_dim=6,
            time_hidden_dim=8,
            num_blocks=2,
        )
    )
    sample = jnp.ones((2, 3, 4), dtype=jnp.float32)
    condition = jnp.ones((2, 3, 8), dtype=jnp.float32)
    flow_head.lazy_init(
        sample,
        jnp.ones((2,), dtype=jnp.float32),
        condition,
        jnp.zeros((2,), dtype=jnp.float32),
        jax.random.key(1),
        train=False,
        rngs=nnx.Rngs(0),
    )

    def loss_fn(model):
        loss, _ = msp_flow_head.meanflow_loss(model, sample, condition, jax.random.key(2), train=True)
        return jnp.mean(loss)

    loss, grads = nnx.value_and_grad(loss_fn)(flow_head)

    assert np.isfinite(np.asarray(loss))
    grad_leaves = jax.tree.leaves(grads)
    assert grad_leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in grad_leaves)
