# -*- coding: utf-8 -*-
"""LoGo token-level dynamic local-global attention layer.

Every token attends locally through a sliding window (SWA); a learned per-token
scalar gate additionally routes tokens through full (global) attention. The two
branches are combined as::

    gate = sigmoid(gate_proj(hidden_states))
    out = (1 - gate) * local_out + gate * global_out

The global branch uses its own q/k/v re-parameterization (``global_qk_param``,
one of ``"scale_offset"`` / ``"linear_proj"``) so it can specialize away from
the local branch while sharing the base projections. Only tokens whose gate
exceeds ``gate_thres`` take the global path (query-only sparsification /
token-level span budget); with the ``triton`` backend this is executed by the
selected-query kernel ``sq_full_attn``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from logo.layers.attn import Attention
from logo.layers.utils import flash_attention_forward
from logo.modules.layernorm import RMSNorm
from logo.modules.rotary import apply_rotary_pos_emb
from logo.ops import sq_full_attn

__all__ = ["LoGoAttention"]


class LoGoAttention(Attention):
    """Token-level dynamic local-global attention."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.local_window_size = config.window_size
        self.global_qk_param = config.global_qk_param

        # Global Q/K/V reparameterization
        if self.global_qk_param == "scale_offset":
            self.q_global_scale = nn.Parameter(torch.ones(self.num_heads * self.head_dim))
            self.q_global_offset = nn.Parameter(torch.zeros(self.num_heads * self.head_dim))
            self.k_global_scale = nn.Parameter(torch.ones(self.num_key_value_heads * self.head_dim))
            self.k_global_offset = nn.Parameter(torch.zeros(self.num_key_value_heads * self.head_dim))
            self.v_global_scale = nn.Parameter(torch.ones(self.num_key_value_heads * self.head_dim))
            self.v_global_offset = nn.Parameter(torch.zeros(self.num_key_value_heads * self.head_dim))
            self._transform_qkv = self._scale_offset_qkv
        elif self.global_qk_param == "linear_proj":
            self.q_global_proj = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
            self.k_global_proj = nn.Parameter(torch.empty(self.num_key_value_heads, self.head_dim, self.head_dim))
            self.v_global_proj = nn.Parameter(torch.empty(self.num_key_value_heads, self.head_dim, self.head_dim))
            for proj, n_heads in (
                (self.q_global_proj, self.num_heads),
                (self.k_global_proj, self.num_key_value_heads),
                (self.v_global_proj, self.num_key_value_heads),
            ):
                nn.init.eye_(proj[0])
                proj.data.copy_(proj[0].unsqueeze(0).repeat(n_heads, 1, 1))
            self._transform_qkv = self._linear_proj_qkv

        self.gate_proj = nn.Linear(self.hidden_size, 1, bias=False)

        self.use_context_norm = config.use_context_norm
        if self.use_context_norm:
            self.context_norm_local = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.context_norm_global = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sparse_full_attn_backend = config.sparse_full_attn_backend
        self.register_buffer("gate_thres", torch.full((1,), config.gate_thres_init))
        self.register_buffer("gate_avg", torch.zeros(1), persistent=False)
        self.register_buffer("global_token_count", torch.zeros(1), persistent=False)
        self.register_buffer("total_token_count", torch.zeros(1), persistent=False)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _scale_offset_qkv(self, q, k, v):
        q = q * self.q_global_scale.view(1, 1, -1) + self.q_global_offset.view(1, 1, -1)
        k = k * self.k_global_scale.view(1, 1, -1) + self.k_global_offset.view(1, 1, -1)
        v = v * self.v_global_scale.view(1, 1, -1) + self.v_global_offset.view(1, 1, -1)
        return q, k, v

    def _linear_proj_qkv(self, q, k, v):
        bsz, q_len, _ = q.size()
        q_view = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k_view = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v_view = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        q_out = torch.einsum("b l h d, h d e -> b l h e", q_view, self.q_global_proj)
        k_out = torch.einsum("b l h d, h d e -> b l h e", k_view, self.k_global_proj)
        v_out = torch.einsum("b l h d, h d e -> b l h e", v_view, self.v_global_proj)
        return q_out.reshape(bsz, q_len, -1), k_out.reshape(bsz, q_len, -1), v_out.reshape(bsz, q_len, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        global_query_states, global_key_states, global_value_states = self._transform_qkv(
            query_states, key_states, value_states
        )

        # reshape to [batch, num_heads, seq_len, head_dim] for RoPE / cache
        query_states, global_query_states = [
            x.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            for x in (query_states, global_query_states)
        ]
        key_states, value_states, global_key_states, global_value_states = [
            x.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            for x in (key_states, value_states, global_key_states, global_value_states)
        ]

        gate = torch.sigmoid(self.gate_proj(hidden_states))

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        global_query_states, global_key_states = apply_rotary_pos_emb(
            global_query_states, global_key_states, cos, sin
        )

        if past_key_value is not None:
            state = past_key_value.update(
                attn_state=(key_states, value_states),
                attn_state_full=(global_key_states, global_value_states),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs={"window_size": self.local_window_size},
            )
            key_states, value_states = state["attn_state"]
            global_key_states, global_value_states = state["attn_state_full"]

        # flash-attention expects [batch, seq_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        global_query_states = global_query_states.transpose(1, 2).contiguous()
        global_key_states = global_key_states.transpose(1, 2).contiguous()
        global_value_states = global_value_states.transpose(1, 2).contiguous()

        dropout_rate = self.attention_dropout if self.training else 0.0

        # Local sliding-window attention
        local_attention_mask = attention_mask
        if local_attention_mask is not None and local_attention_mask.shape[-1] != key_states.shape[1]:
            local_attention_mask = local_attention_mask[:, -key_states.shape[1]:]
        local_attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            local_attention_mask,
            q_len,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            dropout=dropout_rate,
            sliding_window=self.local_window_size,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
            max_seqlen=max_seqlen,
        )
        if self.use_context_norm:
            local_attn_output = self.context_norm_local(local_attn_output)
        local_attn_output = local_attn_output.reshape(bsz, q_len, -1).contiguous()

        # Gated global attention
        is_decode = past_key_value is not None and q_len == 1
        use_triton = self.sparse_full_attn_backend == "triton" and not is_decode

        if not is_decode:
            if use_triton:
                selection = gate.squeeze(-1) >= self.gate_thres
                if cu_seqlens is not None:
                    global_attn_output = sq_full_attn(
                        global_query_states,
                        global_key_states,
                        global_value_states,
                        selection,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    # Pack an equal-length batch for the varlen Triton kernel.
                    q_p = global_query_states.reshape(1, bsz * q_len, self.num_heads, self.head_dim)
                    k_p = global_key_states.reshape(1, bsz * q_len, self.num_key_value_heads, self.head_dim)
                    v_p = global_value_states.reshape(1, bsz * q_len, self.num_key_value_heads, self.head_dim)
                    sel_p = selection.reshape(1, bsz * q_len)
                    cu_p = torch.arange(
                        0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=global_query_states.device
                    )
                    o_p = sq_full_attn(q_p, k_p, v_p, sel_p, cu_seqlens=cu_p)
                    global_attn_output = o_p.reshape(bsz, q_len, self.num_heads, self.head_dim)
            else:
                global_attn_output = flash_attention_forward(
                    global_query_states,
                    global_key_states,
                    global_value_states,
                    attention_mask,
                    q_len,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    dropout=dropout_rate,
                    sliding_window=None,
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                    is_causal=self.is_causal,
                    max_seqlen=max_seqlen,
                )
            if self.use_context_norm:
                global_attn_output = self.context_norm_global(global_attn_output)
            global_attn_output = global_attn_output.reshape(bsz, q_len, -1).contiguous()
        else:
            active = gate >= self.gate_thres
            if active.any():
                global_attn_output = flash_attention_forward(
                    global_query_states,
                    global_key_states,
                    global_value_states,
                    attention_mask,
                    q_len,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    dropout=dropout_rate,
                    sliding_window=None,
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                    is_causal=self.is_causal,
                    max_seqlen=max_seqlen,
                )
                if self.use_context_norm:
                    global_attn_output = self.context_norm_global(global_attn_output)
                global_attn_output = global_attn_output.reshape(bsz, q_len, -1).contiguous()
            else:
                global_attn_output = global_query_states.new_zeros(bsz, q_len, self.num_heads * self.head_dim)

        if self.training:
            with torch.no_grad():
                self.gate_avg.add_(gate.sum())
                self.global_token_count.add_((gate >= self.gate_thres).sum())
                self.total_token_count.add_(gate.numel())
        gate = gate.masked_fill(gate < self.gate_thres, 0.0)

        attn_output = (1 - gate) * local_attn_output + gate * global_attn_output
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None, past_key_value
