from __future__ import annotations

import dataclasses
import math

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


_PYTORCH_NORM_EPS = 1e-5


def _uniform_initializer(bound: float):
    def init(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def _pytorch_default_kernel_init(
    key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    fan_in = math.prod(shape[:-1])
    return _uniform_initializer(1.0 / math.sqrt(fan_in))(key, shape, dtype)


def _pytorch_mha_qkv_kernel_init(
    key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    # PyTorch initializes one combined [3D, D] in-projection with Xavier
    # uniform, so each separated Q/K/V matrix uses variance 0.5 / D.
    fan_in = shape[0]
    bound = math.sqrt(1.5 / fan_in)
    return _uniform_initializer(bound)(key, shape, dtype)


def _pytorch_dense(features: int, input_features: int, *, name: str | None = None) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=_pytorch_default_kernel_init,
        bias_init=_uniform_initializer(1.0 / math.sqrt(input_features)),
        name=name,
    )


def _mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))


def _num_groups(channels: int, requested: int) -> int:
    return math.gcd(channels, requested)


def _sinusoidal_positions(length: int, dim: int, dtype: jnp.dtype) -> jnp.ndarray:
    padded_dim = math.ceil(dim / 2) * 2
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, padded_dim, 2, dtype=jnp.float32) / padded_dim))
    phase = position * inv_freq
    emb = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1).reshape(length, padded_dim)
    return emb[:, :dim].astype(dtype)


def _causal_mask(batch_size: int, length: int) -> jnp.ndarray:
    return nn.make_causal_mask(jnp.ones((batch_size, length), dtype=jnp.bool_))


def _full_attention_mask(batch_size: int, query_length: int, key_length: int) -> jnp.ndarray:
    return jnp.ones((batch_size, 1, query_length, key_length), dtype=jnp.bool_)


class CausalConv1D(nn.Module):
    features: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        return nn.Conv(
            self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="VALID",
            kernel_init=_pytorch_default_kernel_init,
            bias_init=_uniform_initializer(1.0 / math.sqrt(self.kernel_size * x.shape[-1])),
        )(x)


class Conv1DBlock(nn.Module):
    features: int
    kernel_size: int
    stride: int
    num_groups: int = 4
    causal: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.causal:
            x = CausalConv1D(self.features, self.kernel_size, self.stride)(x)
        else:
            x = nn.Conv(
                self.features,
                kernel_size=(self.kernel_size,),
                strides=(self.stride,),
                padding=((self.kernel_size // 2, self.kernel_size // 2),),
                kernel_init=_pytorch_default_kernel_init,
                bias_init=_uniform_initializer(1.0 / math.sqrt(self.kernel_size * x.shape[-1])),
            )(x)
        x = nn.GroupNorm(
            num_groups=_num_groups(self.features, self.num_groups),
            epsilon=_PYTORCH_NORM_EPS,
        )(x)
        return _mish(x)


class ResidualTemporalBlock(nn.Module):
    features: int
    kernel_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    num_groups: int = 8
    causal: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for kernel_size, stride in zip(self.kernel_sizes, self.strides, strict=True):
            x = Conv1DBlock(
                self.features,
                kernel_size,
                stride,
                num_groups=self.num_groups,
                causal=self.causal,
            )(x)
        return x


class TransformerEncoderBlock(nn.Module):
    width: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, mask: jnp.ndarray | None, train: bool) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm(epsilon=_PYTORCH_NORM_EPS)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            deterministic=not train,
            kernel_init=_pytorch_mha_qkv_kernel_init,
            out_kernel_init=_pytorch_default_kernel_init,
            bias_init=nn.initializers.zeros_init(),
            out_bias_init=nn.initializers.zeros_init(),
        )(x, x, x, mask=mask)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = residual + x

        residual = x
        x = nn.LayerNorm(epsilon=_PYTORCH_NORM_EPS)(x)
        x = _pytorch_dense(4 * self.width, self.width)(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = _pytorch_dense(self.width, 4 * self.width)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        return residual + x


class TransformerDecoderBlock(nn.Module):
    width: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        memory: jnp.ndarray,
        *,
        self_mask: jnp.ndarray | None,
        cross_mask: jnp.ndarray | None,
        train: bool,
    ) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm(epsilon=_PYTORCH_NORM_EPS)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            deterministic=not train,
            kernel_init=_pytorch_mha_qkv_kernel_init,
            out_kernel_init=_pytorch_default_kernel_init,
            bias_init=nn.initializers.zeros_init(),
            out_bias_init=nn.initializers.zeros_init(),
        )(x, x, x, mask=self_mask)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = residual + x

        residual = x
        x = nn.LayerNorm(epsilon=_PYTORCH_NORM_EPS)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            deterministic=not train,
            kernel_init=_pytorch_mha_qkv_kernel_init,
            out_kernel_init=_pytorch_default_kernel_init,
            bias_init=nn.initializers.zeros_init(),
            out_bias_init=nn.initializers.zeros_init(),
        )(x, memory, memory, mask=cross_mask)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = residual + x

        residual = x
        x = nn.LayerNorm(epsilon=_PYTORCH_NORM_EPS)(x)
        x = _pytorch_dense(4 * self.width, self.width)(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = _pytorch_dense(self.width, 4 * self.width)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        return residual + x


class ActionVAELinen(nn.Module):
    action_dim: int
    encoder_dim: int = 128
    decoder_dim: int = 128
    skill_block_size: int = 32
    downsample_factor: int = 4
    attn_pdrop: float = 0.1
    use_causal_encoder: bool = True
    use_causal_decoder: bool = True
    encoder_heads: int = 2
    encoder_layers: int = 2
    decoder_heads: int = 4
    decoder_layers: int = 4
    latent_dim: int = 16

    def _conv_schedule(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.downsample_factor == 1:
            return (3, 2), (1, 1)
        num_downsample_layers = int(math.log2(self.downsample_factor))
        return (5, *([3] * num_downsample_layers)), (*([2] * num_downsample_layers), 1)

    @nn.compact
    def encode(self, actions: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = _pytorch_dense(self.encoder_dim, self.action_dim, name="action_proj")(actions)
        kernel_sizes, strides = self._conv_schedule()
        x = ResidualTemporalBlock(
            self.encoder_dim,
            kernel_sizes=kernel_sizes,
            strides=strides,
            causal=self.use_causal_encoder,
            name="conv_block",
        )(x)
        x = x + _sinusoidal_positions(x.shape[1], self.encoder_dim, x.dtype)[None, :, :]

        mask = _causal_mask(x.shape[0], x.shape[1]) if self.use_causal_encoder else None
        for i in range(self.encoder_layers):
            x = TransformerEncoderBlock(
                self.encoder_dim,
                self.encoder_heads,
                self.attn_pdrop,
                name=f"encoder_block_{i}",
            )(x, mask=mask, train=train)
        return x

    @nn.compact
    def decode(self, codes: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = jnp.zeros((codes.shape[0], self.skill_block_size, self.decoder_dim), dtype=codes.dtype)
        x = x + _sinusoidal_positions(self.skill_block_size, self.decoder_dim, x.dtype)[None, :, :]

        self_mask = _causal_mask(x.shape[0], x.shape[1]) if self.use_causal_decoder else None
        cross_mask = _full_attention_mask(x.shape[0], x.shape[1], codes.shape[1])
        for i in range(self.decoder_layers):
            x = TransformerDecoderBlock(
                self.decoder_dim,
                self.decoder_heads,
                self.attn_pdrop,
                name=f"decoder_block_{i}",
            )(x, codes, self_mask=self_mask, cross_mask=cross_mask, train=train)
        return _pytorch_dense(self.action_dim, self.decoder_dim, name="action_head")(x)

    @nn.compact
    def get_sample(self, actions: jnp.ndarray, sample_rng: at.KeyArrayLike, *, train: bool = False) -> jnp.ndarray:
        h = self.encode(actions, train=train)
        moments = _pytorch_dense(self.latent_dim * 2, self.encoder_dim, name="quant_proj")(h)
        mean, logvar = jnp.split(moments, 2, axis=-1)
        logvar = jnp.clip(logvar, -30.0, 20.0)
        return mean + jnp.exp(0.5 * logvar) * jax.random.normal(sample_rng, mean.shape, dtype=mean.dtype)

    @nn.compact
    def encode_mean(self, actions: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        h = self.encode(actions, train=train)
        moments = _pytorch_dense(self.latent_dim * 2, self.encoder_dim, name="quant_proj")(h)
        mean, _ = jnp.split(moments, 2, axis=-1)
        return mean

    @nn.compact
    def get_action(self, z: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        z = _pytorch_dense(self.decoder_dim, self.latent_dim, name="post_quant_proj")(z)
        return self.decode(z, train=train)

    @nn.compact
    def __call__(
        self, actions: jnp.ndarray, sample_rng: at.KeyArrayLike, *, train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = self.encode(actions, train=train)
        moments = _pytorch_dense(self.latent_dim * 2, self.encoder_dim, name="quant_proj")(h)
        mean, logvar = jnp.split(moments, 2, axis=-1)
        logvar = jnp.clip(logvar, -30.0, 20.0)
        z = mean + jnp.exp(0.5 * logvar) * jax.random.normal(sample_rng, mean.shape, dtype=mean.dtype)
        recon = self.get_action(z, train=train)
        return recon, mean, logvar


class MspActionVAE(_model.BaseModel):
    def __init__(self, config: pi0_config.MspActionVAEConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = dataclasses.replace(config)
        self.kl_weight = config.kl_weight
        self.action_vae = nnx_bridge.ToNNX(
            ActionVAELinen(
                action_dim=config.action_dim,
                encoder_dim=config.encoder_dim,
                decoder_dim=config.decoder_dim,
                skill_block_size=config.action_horizon,
                downsample_factor=config.downsample_factor,
                attn_pdrop=config.attn_pdrop,
                use_causal_encoder=config.use_causal_encoder,
                use_causal_decoder=config.use_causal_decoder,
                encoder_heads=config.encoder_heads,
                encoder_layers=config.encoder_layers,
                decoder_heads=config.decoder_heads,
                decoder_layers=config.decoder_layers,
                latent_dim=config.latent_dim,
            )
        )
        self.action_vae.lazy_init(
            jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32),
            jax.random.key(0),
            train=False,
            rngs=rngs,
        )
        self.deterministic = True

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        del observation
        sample_rng, dropout_rng = jax.random.split(rng)
        recon, mean, logvar = self.action_vae(actions, sample_rng, train=train, rngs=nnx.Rngs(dropout=dropout_rng))
        recon_loss = jnp.mean(jnp.abs(recon - actions), axis=-1)
        kl_loss = 0.5 * jnp.sum(jnp.square(mean) + jnp.exp(logvar) - 1.0 - logvar, axis=(1, 2))
        return recon_loss + self.kl_weight * kl_loss[:, None]

    @override
    def compute_loss_with_info(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        del observation
        sample_rng, dropout_rng = jax.random.split(rng)
        recon, mean, logvar = self.action_vae(actions, sample_rng, train=train, rngs=nnx.Rngs(dropout=dropout_rng))
        recon_loss = jnp.mean(jnp.abs(recon - actions), axis=-1)
        kl_loss = 0.5 * jnp.sum(jnp.square(mean) + jnp.exp(logvar) - 1.0 - logvar, axis=(1, 2))
        weighted_kl_loss = self.kl_weight * kl_loss[:, None]
        total_loss = recon_loss + weighted_kl_loss
        return total_loss, {
            "recon_loss": jnp.mean(recon_loss),
            "kl_loss": jnp.mean(weighted_kl_loss),
        }

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        del kwargs
        batch_size = observation.state.shape[0]
        latent_horizon = math.ceil(self.action_horizon / self.config.downsample_factor)
        z = jax.random.normal(rng, (batch_size, latent_horizon, self.config.latent_dim), dtype=jnp.float32)
        return self.action_vae(z, train=False, method="get_action")
