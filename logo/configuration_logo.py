# -*- coding: utf-8 -*-
"""Configuration for LoGo token-level dynamic local-global attention."""

from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

__all__ = ["LoGoConfig"]


class LoGoConfig(PretrainedConfig):
    r"""Configuration for the LoGo token-level dynamic local-global attention model.

    LoGo runs every token through sliding-window (local) attention, while a
    learned per-token scalar gate additionally routes selected tokens through
    full (global) attention. See ``logo.layers.LoGoAttention`` for details.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads for Grouped Query Attention. Defaults to
            `num_attention_heads` (i.e. standard Multi-Head Attention).
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length the model might be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated-normal weight initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon used by the RMSNorm layers.
        layer_norm_eps (`float`, *optional*):
            If set, use `nn.LayerNorm` (with this epsilon) instead of RMSNorm.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/value states.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration; see `transformers` RoPE utilities. Valid
            `rope_type` values are `default`, `linear`, `dynamic`, `yarn`,
            `longrope` and `llama3`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the q/k/v projection layers.
        attention_out_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the output projection.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio applied to the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP projection layers.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            Residual dropout probability.
        attn_type_list (`List[int]`, *optional*):
            Per-layer attention type: `0` selects LoGo attention,
            any other value selects standard full attention. Defaults to all
            zeros (every layer is a LoGo layer).
        window_size (`int`, *optional*, defaults to 128):
            Sliding-window size for the local (SWA) branch.
        global_qk_param (`str`, *optional*, defaults to `"scale_offset"`):
            How the global branch re-parameterizes q/k/v relative to the local
            branch. One of `"scale_offset"` (per-channel affine) or
            `"linear_proj"` (per-head mixing matrix).
        sparse_full_attn_backend (`str`, *optional*, defaults to `"flash"`):
            Backend for the global branch: `"flash"` (dense flash-attention) or
            `"triton"` (selected-query kernel that skips unselected tokens).
        gate_thres_init (`float`, *optional*, defaults to 0.5):
            Initial value of the gate threshold. Only tokens whose gate exceeds
            this threshold take the global path (query-only sparsification /
            token-level span budget).
        use_context_norm (`bool`, *optional*, defaults to `True`):
            If `True`, apply a per-branch RMSNorm to the attention context before
            gating.
    """

    model_type = "logo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        layer_norm_eps: Optional[float] = None,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        resid_pdrop: float = 0.0,
        # ---- LoGo-specific knobs ----
        attn_type_list: Optional[List[int]] = None,
        window_size: int = 128,
        global_qk_param: str = "scale_offset",
        sparse_full_attn_backend: str = "flash",
        gate_thres_init: float = 0.5,
        use_context_norm: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.resid_pdrop = resid_pdrop

        # LoGo-specific knobs
        if attn_type_list is None:
            attn_type_list = [0] * num_hidden_layers
        self.attn_type_list = attn_type_list
        self.window_size = window_size
        self.global_qk_param = global_qk_param
        self.sparse_full_attn_backend = sparse_full_attn_backend
        self.gate_thres_init = gate_thres_init
        self.use_context_norm = use_context_norm

        self._validate()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate(self):
        if self.global_qk_param not in ("scale_offset", "linear_proj"):
            raise ValueError(
                f"`global_qk_param` must be 'scale_offset' or 'linear_proj', got {self.global_qk_param!r}"
            )
        if self.sparse_full_attn_backend not in ("flash", "triton"):
            raise ValueError(
                f"`sparse_full_attn_backend` must be 'flash' or 'triton', got {self.sparse_full_attn_backend!r}"
            )
        if len(self.attn_type_list) != self.num_hidden_layers:
            raise ValueError(
                f"`attn_type_list` must have length num_hidden_layers={self.num_hidden_layers}, "
                f"got {len(self.attn_type_list)}"
            )

        # Validate the RoPE arguments. BC: if there is a 'type' field, promote it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
