import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.msp_scale_head as msp_scale_head
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"
    use_msp_action_head: bool = False
    freeze_msp_vae: bool = True
    msp_action_dim: int | None = None
    msp_encoder_dim: int = 128
    msp_decoder_dim: int = 128
    msp_downsample_factor: int = 4
    msp_attn_pdrop: float = 0.1
    msp_encoder_heads: int = 2
    msp_encoder_layers: int = 2
    msp_decoder_heads: int = 4
    msp_decoder_layers: int = 4
    msp_latent_dim: int = 16
    msp_kl_weight: float = 1e-6
    msp_latent_horizon: int | None = None
    msp_scales: tuple[int, ...] | None = None
    msp_flow_num_blocks: int = 3
    msp_flow_hidden_dim: int = 256
    msp_flow_time_dim: int = 32
    msp_flow_time_hidden_dim: int = 256
    msp_flow_dropout_rate: float = 0.1
    msp_flow_ratio: float = 0.5
    msp_flow_lognormal_mean: float = -0.4
    msp_flow_lognormal_std: float = 1.0
    msp_flow_cfg_scale: float = 2.0
    msp_flow_adaptive_gamma: float = 0.5
    msp_flow_adaptive_c: float = 1e-3

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.use_msp_action_head:
            if not self.pi05:
                raise ValueError("MSP action head currently supports only pi0.5 (set pi05=True)")
            if self.msp_action_dim is None:
                object.__setattr__(self, "msp_action_dim", self.action_dim)
            assert self.msp_action_dim is not None
            if self.msp_action_dim > self.action_dim:
                raise ValueError(
                    f"msp_action_dim must be <= action_dim for padding compatibility, got "
                    f"{self.msp_action_dim} > {self.action_dim}"
                )
            latent_horizon = self.msp_latent_horizon
            if latent_horizon is None:
                latent_horizon = msp_scale_head.latent_horizon_from_action_horizon(
                    self.action_horizon, self.msp_downsample_factor
                )
                object.__setattr__(self, "msp_latent_horizon", latent_horizon)
            if self.msp_scales is None:
                object.__setattr__(self, "msp_scales", msp_scale_head.default_msp_scales(latent_horizon))
            assert self.msp_scales is not None
            if self.msp_scales[-1] != latent_horizon:
                raise ValueError(
                    f"msp_scales must end with latent horizon {latent_horizon}, got {self.msp_scales[-1]}"
                )
            if any(a >= b for a, b in zip(self.msp_scales[:-1], self.msp_scales[1:], strict=True)):
                raise ValueError(f"msp_scales must be strictly increasing, got {self.msp_scales}")
            if self.msp_flow_num_blocks < 1:
                raise ValueError("msp_flow_num_blocks must be positive")
            if self.msp_flow_time_dim % 2 != 0:
                raise ValueError("msp_flow_time_dim must be even")
            if not 0.0 <= self.msp_flow_dropout_rate < 1.0:
                raise ValueError("msp_flow_dropout_rate must be in [0, 1)")
            if not 0.0 <= self.msp_flow_ratio <= 1.0:
                raise ValueError("msp_flow_ratio must be in [0, 1]")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if self.use_msp_action_head and self.freeze_msp_vae:
            filters.append(nnx_utils.PathRegex(".*msp_action_vae.*"))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def make_msp_vae_config(self) -> "MspActionVAEConfig":
        assert self.msp_action_dim is not None
        return MspActionVAEConfig(
            action_dim=self.msp_action_dim,
            action_horizon=self.action_horizon,
            encoder_dim=self.msp_encoder_dim,
            decoder_dim=self.msp_decoder_dim,
            downsample_factor=self.msp_downsample_factor,
            attn_pdrop=self.msp_attn_pdrop,
            use_causal_encoder=True,
            use_causal_decoder=True,
            encoder_heads=self.msp_encoder_heads,
            encoder_layers=self.msp_encoder_layers,
            decoder_heads=self.msp_decoder_heads,
            decoder_layers=self.msp_decoder_layers,
            latent_dim=self.msp_latent_dim,
            kl_weight=self.msp_kl_weight,
        )


@dataclasses.dataclass(frozen=True)
class MspActionVAEConfig(_model.BaseModelConfig):
    """Stage-1 MSP action VAE config.

    This model is intentionally observation-free: it trains only on action chunks and does not create or load the pi0.5
    VLM/action-expert stack.
    """

    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 1

    encoder_dim: int = 128
    decoder_dim: int = 128
    downsample_factor: int = 4
    attn_pdrop: float = 0.1
    use_causal_encoder: bool = True
    use_causal_decoder: bool = True
    encoder_heads: int = 2
    encoder_layers: int = 2
    decoder_heads: int = 4
    decoder_layers: int = 4
    latent_dim: int = 16
    kl_weight: float = 1e-6

    def __post_init__(self):
        if self.downsample_factor < 1 or 2 ** int(math.log2(self.downsample_factor)) != self.downsample_factor:
            raise ValueError("downsample_factor must be a positive power of 2")
        if self.encoder_dim % self.encoder_heads != 0:
            raise ValueError("encoder_dim must be divisible by encoder_heads")
        if self.decoder_dim % self.decoder_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.MSP_VAE

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.msp_vae import MspActionVAE

        return MspActionVAE(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
