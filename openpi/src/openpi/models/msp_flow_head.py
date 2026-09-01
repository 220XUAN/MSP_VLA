"""JAX port of MSP's scale-wise MeanFlow action head."""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def _xavier_dense(features: int, *, name: str) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros_init(),
        name=name,
    )


def _dropout(x: jnp.ndarray, rng: jax.Array, *, rate: float, train: bool) -> jnp.ndarray:
    if not train or rate == 0.0:
        return x
    keep_prob = 1.0 - rate
    keep = jax.random.bernoulli(rng, keep_prob, x.shape)
    return jnp.where(keep, x / keep_prob, 0).astype(x.dtype)


class LearnedPosEmb(nn.Module):
    """Learned Fourier time embedding used by the original MSP flow head."""

    output_size: int = 32

    @nn.compact
    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        if self.output_size % 2 != 0:
            raise ValueError(f"output_size must be even, got {self.output_size}")
        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=0.2),
            (self.output_size // 2, 1),
        )
        phase = 2 * jnp.pi * value[:, None] @ kernel.T
        return jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)


class TimeEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _xavier_dense(self.hidden_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.LayerNorm(name="norm")(x)
        return _xavier_dense(self.hidden_dim, name="fc2")(x)


class MlpResNetBlock(nn.Module):
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, dropout_rng: jax.Array, *, train: bool) -> jnp.ndarray:
        residual = x
        x = _dropout(x, dropout_rng, rate=self.dropout_rate, train=train)
        x = nn.LayerNorm(name="norm1")(x)
        x = _xavier_dense(self.hidden_dim * 4, name="dense1")(x)
        x = nn.gelu(x)
        x = _xavier_dense(self.hidden_dim, name="dense2")(x)
        return residual + x


class MspScaleFlowHead(nn.Module):
    """Token-wise `MPScalseFlowhead.model` / `MpScaleMlpResNet` port."""

    latent_dim: int
    condition_dim: int
    hidden_dim: int
    time_dim: int = 32
    time_hidden_dim: int = 256
    num_blocks: int = 3
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        sample: jnp.ndarray,
        timestep: jnp.ndarray,
        condition: jnp.ndarray,
        r: jnp.ndarray,
        dropout_rng: jax.Array,
        *,
        train: bool,
    ) -> jnp.ndarray:
        if sample.ndim != 3 or sample.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected sample [B, N, {self.latent_dim}], got {sample.shape}")
        if condition.shape[:2] != sample.shape[:2] or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition [B, N, {self.condition_dim}] matching sample, got {condition.shape}"
            )

        time_process = LearnedPosEmb(self.time_dim, name="time_process")
        time_encoder = TimeEncoder(self.time_hidden_dim, name="time_encoder")
        time_embedding = time_encoder(time_process(timestep))
        r_embedding = time_encoder(time_process(r))
        time_embedding = jnp.broadcast_to(
            (time_embedding + r_embedding)[:, None, :],
            (*sample.shape[:2], self.time_hidden_dim),
        )

        x = jnp.concatenate([condition, time_embedding.astype(condition.dtype), sample], axis=-1)
        x = _xavier_dense(self.hidden_dim, name="dense1")(x)
        block_rngs = jax.random.split(dropout_rng, self.num_blocks)
        for block_index in range(self.num_blocks):
            x = MlpResNetBlock(
                self.hidden_dim,
                dropout_rate=self.dropout_rate,
                name=f"mlp_res_block_{block_index}",
            )(x, block_rngs[block_index], train=train)
        x = nn.gelu(x)
        return _xavier_dense(self.latent_dim, name="dense2")(x)


def sample_time_pairs(
    rng: jax.Array,
    batch_size: int,
    *,
    flow_ratio: float = 0.5,
    lognormal_mean: float = -0.4,
    lognormal_std: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Port of `MPScalseFlowhead.sample_t_r`."""
    sample_rng, permutation_rng = jax.random.split(rng)
    samples = jax.nn.sigmoid(
        jax.random.normal(sample_rng, (batch_size, 2), dtype=jnp.float32) * lognormal_std + lognormal_mean
    )
    t = jnp.max(samples, axis=-1)
    r = jnp.min(samples, axis=-1)
    num_equal_pairs = int(flow_ratio * batch_size)
    equal_indices = jax.random.permutation(permutation_rng, batch_size)[:num_equal_pairs]
    r = r.at[equal_indices].set(t[equal_indices])
    return t, r


def adaptive_l2_loss(error: jnp.ndarray, *, gamma: float = 0.5, c: float = 1e-3) -> jnp.ndarray:
    """Per-example form of MSP's adaptive L2; averaging it matches the PyTorch loss."""
    delta_sq = jnp.mean(jnp.square(error), axis=tuple(range(1, error.ndim)))
    weight = jax.lax.stop_gradient(1.0 / jnp.power(delta_sq + c, 1.0 - gamma))
    return weight * delta_sq


def meanflow_loss(
    model: Callable[..., jnp.ndarray],
    target: jnp.ndarray,
    condition: jnp.ndarray,
    rng: jax.Array,
    *,
    flow_ratio: float = 0.5,
    lognormal_mean: float = -0.4,
    lognormal_std: float = 1.0,
    cfg_scale: float = 2.0,
    adaptive_gamma: float = 0.5,
    adaptive_c: float = 1e-3,
    train: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute MSP's JVP MeanFlow objective for one scale block."""
    time_rng, noise_rng, guide_dropout_rng, model_dropout_rng = jax.random.split(rng, 4)
    t, r = sample_time_pairs(
        time_rng,
        target.shape[0],
        flow_ratio=flow_ratio,
        lognormal_mean=lognormal_mean,
        lognormal_std=lognormal_std,
    )
    t_expanded = t[:, None, None]
    r_expanded = r[:, None, None]
    noise = jax.random.normal(noise_rng, target.shape, dtype=target.dtype)
    z = (1.0 - t_expanded) * target + t_expanded * noise
    velocity = noise - target

    guide_velocity = model(z, t, condition, t, guide_dropout_rng, train=train)
    guide_velocity = jax.lax.stop_gradient(guide_velocity)
    v_hat = cfg_scale * velocity + (1.0 - cfg_scale) * guide_velocity

    def model_fn(z_value: jnp.ndarray, t_value: jnp.ndarray, r_value: jnp.ndarray) -> jnp.ndarray:
        return model(z_value, t_value, condition, r_value, model_dropout_rng, train=train)

    prediction, derivative = jax.jvp(
        model_fn,
        (z, t, r),
        (v_hat, jnp.ones_like(t), jnp.zeros_like(r)),
    )
    target_velocity = v_hat - (t_expanded - r_expanded) * derivative
    error = prediction - jax.lax.stop_gradient(target_velocity)
    return adaptive_l2_loss(error, gamma=adaptive_gamma, c=adaptive_c), jnp.mean(
        jnp.square(jax.lax.stop_gradient(error)), axis=tuple(range(1, error.ndim))
    )


def generate(
    model: Callable[..., jnp.ndarray], condition: jnp.ndarray, sample: jnp.ndarray
) -> jnp.ndarray:
    """Original MSP one-step generation: x_0 = x_1 - u(x_1, 1, 0)."""
    batch_size = condition.shape[0]
    timestep = jnp.ones((batch_size,), dtype=jnp.float32)
    r = jnp.zeros((batch_size,), dtype=jnp.float32)
    velocity = model(sample, timestep, condition, r, jax.random.key(0), train=False)
    return sample - velocity
