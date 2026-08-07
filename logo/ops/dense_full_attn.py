# -*- coding: utf-8 -*-
# Dense reference for the LoGo selected-query full-attention operator.
#
# `sq_full_attn_dense_ref` runs the SAME Triton kernels as the optimized op but
# with a full (all-ones) selection, so every token is a "selected" query. This
# reproduces standard full causal attention while sharing the exact kernel
# binary / accumulation order, making selected-row bitwise comparison auditable.
#
# `sdpa_causal_ref` is an independent absolute-correctness reference based on
# torch SDPA (not bitwise; used for a sanity cross-check).

from typing import Optional

import torch
import torch.nn.functional as F

from .selected_full_attn import sq_full_attn


def sq_full_attn_dense_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full causal attention via the shared kernels (selection = all-ones)."""
    B, T, HQ, K = q.shape
    selection = torch.ones(B, T, dtype=torch.bool, device=q.device)
    return sq_full_attn(q, k, v, selection, scale=scale, cu_seqlens=cu_seqlens)


def _repeat_kv(x: torch.Tensor, g: int) -> torch.Tensor:
    # x: [B, T, H, D] -> [B, T, H*g, D]
    B, T, H, D = x.shape
    return x[:, :, :, None, :].expand(B, T, H, g, D).reshape(B, T, H * g, D)


def sdpa_causal_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Independent causal-attention reference using torch SDPA.

    q: [B,T,HQ,K], k/v: [B,T,H,*]. Returns [B,T,HQ,V].
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5

    kk = _repeat_kv(k, G)
    vv = _repeat_kv(v, G)
    qh = q.transpose(1, 2)   # [B,HQ,T,K]
    kh = kk.transpose(1, 2)
    vh = vv.transpose(1, 2)

    if cu_seqlens is None:
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale)
        return o.transpose(1, 2).contiguous()

    # varlen (B==1): block-diagonal causal mask per sequence
    assert B == 1
    cu = cu_seqlens.tolist()
    outs = []
    for s in range(len(cu) - 1):
        bos, eos = cu[s], cu[s + 1]
        o = F.scaled_dot_product_attention(
            qh[:, :, bos:eos], kh[:, :, bos:eos], vh[:, :, bos:eos],
            is_causal=True, scale=scale,
        )
        outs.append(o)
    o = torch.cat(outs, dim=2)
    return o.transpose(1, 2).contiguous()
