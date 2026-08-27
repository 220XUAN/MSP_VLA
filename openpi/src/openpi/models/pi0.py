import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.msp_scale_head as msp_scale_head
from openpi.models.msp_vae import ActionVAELinen
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def token_posemb_sincos(
    pos: at.Real[at.Array, " s"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "s {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_msp_action_head = config.use_msp_action_head
        self.msp_scales = config.msp_scales
        self.msp_action_dim = config.msp_action_dim if config.msp_action_dim is not None else config.action_dim
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.use_msp_action_head:
            assert self.msp_scales is not None
            msp_vae_config = config.make_msp_vae_config()
            total_msp_tokens = sum(self.msp_scales)
            self.msp_action_vae = nnx_bridge.ToNNX(
                ActionVAELinen(
                    action_dim=msp_vae_config.action_dim,
                    encoder_dim=msp_vae_config.encoder_dim,
                    decoder_dim=msp_vae_config.decoder_dim,
                    skill_block_size=msp_vae_config.action_horizon,
                    downsample_factor=msp_vae_config.downsample_factor,
                    attn_pdrop=msp_vae_config.attn_pdrop,
                    use_causal_encoder=msp_vae_config.use_causal_encoder,
                    use_causal_decoder=msp_vae_config.use_causal_decoder,
                    encoder_heads=msp_vae_config.encoder_heads,
                    encoder_layers=msp_vae_config.encoder_layers,
                    decoder_heads=msp_vae_config.decoder_heads,
                    decoder_layers=msp_vae_config.decoder_layers,
                    latent_dim=msp_vae_config.latent_dim,
                )
            )
            self.msp_action_vae.lazy_init(
                jnp.ones((1, config.action_horizon, self.msp_action_dim), dtype=jnp.float32),
                jax.random.key(0),
                train=False,
                rngs=rngs,
            )
            self.msp_latent_in_proj = nnx.Linear(config.msp_latent_dim, action_expert_config.width, rngs=rngs)
            self.msp_latent_out_proj = nnx.Linear(action_expert_config.width, config.msp_latent_dim, rngs=rngs)
            self.msp_scale_embed = nnx.Embed(
                num_embeddings=len(self.msp_scales),
                features=action_expert_config.width,
                rngs=rngs,
            )
            init_std = 0.02
            pos_shape = (1, total_msp_tokens, action_expert_config.width)
            self.msp_encoder_pos_embed = nnx.Param(jax.random.normal(rngs.params(), pos_shape, dtype=jnp.float32) * init_std)
            self.msp_decoder_pos_embed = nnx.Param(jax.random.normal(rngs.params(), pos_shape, dtype=jnp.float32) * init_std)
            self.msp_diffusion_pos_embed = nnx.Param(
                jax.random.normal(rngs.params(), pos_shape, dtype=jnp.float32) * init_std
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _msp_pos_slice(self, pos_embed: nnx.Param, *, block_index: int | None, dtype: jnp.dtype) -> jnp.ndarray:
        assert self.msp_scales is not None
        pos = msp_scale_head.slice_scale_positions(pos_embed.value, self.msp_scales, block_index)
        return pos.astype(dtype)

    def _embed_msp_suffix(
        self,
        latent_inputs: jnp.ndarray,
        scales: tuple[int, ...],
        *,
        block_index: int | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"],
        at.Int[at.Array, "b s"],
    ]:
        scale_ids = msp_scale_head.build_scale_ids(scales)
        action_tokens = self.msp_latent_in_proj(latent_inputs)
        scale_emb = self.msp_scale_embed(scale_ids)[None, :, :]
        encoder_pos = self._msp_pos_slice(
            self.msp_encoder_pos_embed,
            block_index=block_index,
            dtype=action_tokens.dtype,
        )
        decoder_pos = self._msp_pos_slice(
            self.msp_decoder_pos_embed,
            block_index=block_index,
            dtype=action_tokens.dtype,
        )
        tokens = action_tokens + scale_emb + encoder_pos + decoder_pos
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        # Keep global `positions` for sequence/cache layout, but override RoPE with
        # per-scale local positions so each MSP block resets to 0..scale_len-1.
        rope_positions = msp_scale_head.build_block_local_positions(scales, batch_size=tokens.shape[0])
        adarms_cond = jnp.zeros((tokens.shape[0], self.msp_latent_in_proj.out_features), dtype=tokens.dtype)
        return tokens, input_mask, adarms_cond, rope_positions

    def _select_msp_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        return actions[..., : self.msp_action_dim]

    def _pad_msp_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        if actions.shape[-1] == self.action_dim:
            return actions
        pad_width = ((0, 0),) * (actions.ndim - 1) + ((0, self.action_dim - actions.shape[-1]),)
        return jnp.pad(actions, pad_width)

    def _compute_msp_loss(
        self, preprocess_rng: at.KeyArrayLike | None, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._compute_msp_loss_with_info(preprocess_rng, observation, actions, train=train)
        return loss

    def _compute_msp_loss_with_info(
        self, preprocess_rng: at.KeyArrayLike | None, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        assert self.msp_scales is not None
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        finest_latent = self.msp_action_vae(self._select_msp_actions(actions), train=False, method="encode_mean")
        input_latents, target_latents = msp_scale_head.build_teacher_forced_inputs(finest_latent, self.msp_scales)
        total_suffix_len = target_latents.shape[1]

        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, adarms_cond, rope_positions = self._embed_msp_suffix(
            input_latents,
            self.msp_scales,
            block_index=None,
        )
        suffix_attn_mask = msp_scale_head.build_suffix_block_attention_mask(
            self.msp_scales, batch_size=suffix_tokens.shape[0], input_mask=suffix_mask
        )
        attn_mask = msp_scale_head.build_full_attention_mask(prefix_mask, suffix_mask, suffix_attn_mask)
        positions = jnp.cumsum(jnp.concatenate([prefix_mask, suffix_mask], axis=1), axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            rope_positions=[None, rope_positions],
        )
        diffusion_pos = self._msp_pos_slice(
            self.msp_diffusion_pos_embed,
            block_index=None,
            dtype=suffix_out.dtype,
        )
        assert diffusion_pos.shape[1] == total_suffix_len
        pred_latents = self.msp_latent_out_proj(suffix_out[:, -total_suffix_len:] + diffusion_pos)
        token_loss = jnp.mean(jnp.square(pred_latents - target_latents), axis=-1)

        scale_starts, scale_ends = msp_scale_head.build_scale_segment_bounds(self.msp_scales)
        max_scale = float(self.msp_scales[-1])
        weighted_block_losses = []
        metric_info = {}
        for block_idx, scale in enumerate(self.msp_scales):
            start = int(scale_starts[block_idx])
            end = int(scale_ends[block_idx])
            block_loss = jnp.mean(token_loss[:, start:end], axis=-1)
            weighted_block_loss = block_loss * (scale / max_scale)
            weighted_block_losses.append(weighted_block_loss)
            metric_info[f"msp_loss_scale_{scale}"] = jnp.mean(block_loss)
            metric_info[f"msp_loss_scale_{scale}_weighted"] = jnp.mean(weighted_block_loss)

        weighted_total = jnp.sum(jnp.stack(weighted_block_losses, axis=-1), axis=-1)
        finest_start = int(scale_starts[-1])
        finest_end = int(scale_ends[-1])
        finest_block_loss = jnp.mean(token_loss[:, finest_start:finest_end], axis=-1)
        metric_info["msp_loss_total"] = jnp.mean(weighted_total)
        metric_info["msp_loss_finest"] = jnp.mean(finest_block_loss)
        metric_info["msp_loss_finest_weighted"] = jnp.mean(weighted_block_losses[-1])

        return weighted_total[:, None], metric_info

    def _sample_msp_actions(self, observation: _model.Observation) -> _model.Actions:
        assert self.msp_scales is not None
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        latent_dim = self.msp_latent_out_proj.out_features
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
        scale_starts, scale_ends = msp_scale_head.build_scale_segment_bounds(self.msp_scales)

        generated_blocks: list[jnp.ndarray] = []
        for block_idx in range(len(self.msp_scales)):
            current_scale = (self.msp_scales[block_idx],)
            block_start = int(scale_starts[block_idx])
            block_end = int(scale_ends[block_idx])
            latent_inputs = msp_scale_head.build_current_scale_inputs(
                generated_blocks,
                self.msp_scales,
                batch_size=batch_size,
                latent_dim=latent_dim,
                dtype=prefix_tokens.dtype,
            )
            suffix_tokens, suffix_mask, adarms_cond, rope_positions = self._embed_msp_suffix(
                latent_inputs,
                current_scale,
                block_index=block_idx,
            )
            suffix_attn_mask = msp_scale_head.build_current_block_attention_mask(
                self.msp_scales,
                block_idx,
                batch_size=batch_size,
            )
            prefix_attn_mask = jnp.broadcast_to(prefix_mask[:, None, :], (batch_size, current_scale[0], prefix_mask.shape[1]))
            attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            start = prefix_mask.shape[1] + block_start
            positions = jnp.broadcast_to(
                jnp.arange(start, start + current_scale[0], dtype=jnp.int32)[None, :],
                (batch_size, current_scale[0]),
            )
            (_, suffix_out), kv_cache = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                rope_positions=[None, rope_positions],
            )
            diffusion_pos = self._msp_pos_slice(
                self.msp_diffusion_pos_embed,
                block_index=block_idx,
                dtype=suffix_out.dtype,
            )
            assert diffusion_pos.shape[1] == current_scale[0]
            pred_latents = self.msp_latent_out_proj(suffix_out[:, -current_scale[0] :] + diffusion_pos)
            generated_blocks.append(pred_latents)

        decoded_actions = self.msp_action_vae(generated_blocks[-1], train=False, method="get_action")
        return self._pad_msp_actions(decoded_actions)

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.use_msp_action_head:
            preprocess_rng, _ = jax.random.split(rng)
            return self._compute_msp_loss(preprocess_rng, observation, actions, train=train)
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss_with_info(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        if self.use_msp_action_head:
            preprocess_rng, _ = jax.random.split(rng)
            return self._compute_msp_loss_with_info(preprocess_rng, observation, actions, train=train)
        return super().compute_loss_with_info(rng, observation, actions, train=train)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if self.use_msp_action_head:
            del rng, num_steps, noise
            return self._sample_msp_actions(observation)
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
