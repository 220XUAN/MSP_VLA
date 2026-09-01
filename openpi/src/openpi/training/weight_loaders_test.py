import re

import flax.traverse_util
import numpy as np

from openpi.training import weight_loaders


def test_msp_missing_regex_initializes_only_new_head_and_regular_action_expert_norms():
    pattern = re.compile(weight_loaders.MSP_ACTION_EXPERT_MISSING_REGEX)

    assert pattern.fullmatch("PaliGemma/llm/layers/pre_attention_norm_1/scale")
    assert pattern.fullmatch("PaliGemma/llm/layers/pre_ffw_norm_1/scale")
    assert pattern.fullmatch("PaliGemma/llm/final_norm_1/scale")
    assert pattern.fullmatch("msp_latent_in_proj/kernel")
    assert not pattern.fullmatch("PaliGemma/llm/layers/attn_1/q_einsum/kernel")


def test_merge_params_keeps_compatible_expert_weights_and_initializes_regular_norms():
    reference = flax.traverse_util.unflatten_dict(
        {
            "PaliGemma/llm/layers/attn_1/kernel": np.zeros((2, 2), dtype=np.float32),
            "PaliGemma/llm/layers/pre_attention_norm_1/scale": np.zeros((2,), dtype=np.float32),
            "msp_latent_in_proj/kernel": np.zeros((2, 2), dtype=np.float32),
        },
        sep="/",
    )
    checkpoint = flax.traverse_util.unflatten_dict(
        {
            "PaliGemma/llm/layers/attn_1/kernel": np.ones((2, 2), dtype=np.float32),
            "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel": np.ones((2, 6), dtype=np.float32),
        },
        sep="/",
    )

    merged = weight_loaders._merge_params(  # noqa: SLF001
        checkpoint,
        reference,
        missing_regex=weight_loaders.MSP_ACTION_EXPERT_MISSING_REGEX,
    )
    flat = flax.traverse_util.flatten_dict(merged, sep="/")

    assert np.array_equal(flat["PaliGemma/llm/layers/attn_1/kernel"], np.ones((2, 2), dtype=np.float32))
    assert np.array_equal(
        flat["PaliGemma/llm/layers/pre_attention_norm_1/scale"],
        np.zeros((2,), dtype=np.float32),
    )
    assert np.array_equal(flat["msp_latent_in_proj/kernel"], np.zeros((2, 2), dtype=np.float32))
    assert "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel" not in flat
