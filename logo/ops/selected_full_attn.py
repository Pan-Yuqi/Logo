# -*- coding: utf-8 -*-
"""Selected-query full causal attention implemented in Triton.

Selected queries are packed into a varlen axis while keys and values retain their
full sequence layout. Causal masking uses each query's original position.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from einops import reduce

from .common import (
    SelInfo,
    build_selection,
    get_bwd_config,
    get_fwd_config,
    prepare_chunk_indices,
)
from .triton_utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    contiguous,
    exp2,
    log2,
)

RCP_LN2 = 1.4426950216


@triton.jit(do_not_specialize=['Tq_total'])
def sq_full_attn_fwd_kernel(
    q,            # compressed queries [Sq, HQ, K]
    k,            # full keys          [Tk_total, H, K]
    v,            # full values        [Tk_total, H, V]
    o,            # compressed out     [Sq, HQ, V]
    lse,          # compressed lse     [Sq, HQ]
    pos_ids,      # int32 [Sq] real position of each compressed query
    scale,
    cu_seqlens_q,
    cu_seqlens_k,
    chunk_indices,
    Tq_total,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_hq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hq // G
    RCP_LN2: tl.constexpr = 1.4426950216

    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)
    Tq = eos_q - bos_q
    Tk = eos_k - bos_k

    p_q = tl.make_block_ptr(q + (bos_q * HQ + i_hq) * K, (Tq, K), (HQ * K, 1), (i_tl * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos_q * HQ + i_hq) * V, (Tq, V), (HQ * V, 1), (i_tl * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_tl * BT,), (BT,), (0,))

    o_qi = i_tl * BT + tl.arange(0, BT)
    m_q = o_qi < Tq
    b_pos = tl.load(pos_ids + bos_q + o_qi, mask=m_q, other=0).to(tl.int32)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    max_pos = tl.max(tl.where(m_q, b_pos, 0))
    min_pos = tl.min(tl.where(m_q, b_pos, 2147483647))
    hi = (max_pos // BS + 1) * BS
    split = (min_pos // BS) * BS

    # KV blocks below min_pos are fully causal.
    for i_s in range(0, split, BS):
        p_k = tl.make_block_ptr(k + (bos_k * H + i_h) * K, (K, Tk), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos_k * H + i_h) * V, (Tk, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        b_p = exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    # Apply the causal mask in the diagonal region.
    for i_s in range(split, hi, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < Tk
        p_k = tl.make_block_ptr(k + (bos_k * H + i_h) * K, (K, Tk), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos_k * H + i_h) * V, (Tk, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_s = tl.where((b_pos[:, None] >= o_k[None, :]) & m_k[None, :] & m_q[:, None], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        b_p = exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def sq_full_attn_bwd_preprocess_kernel(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V
    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)
    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.jit(do_not_specialize=['Tq_total'])
def sq_full_attn_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    pos_ids,
    scale,
    cu_seqlens_q,
    cu_seqlens_k,
    chunk_indices,
    Tq_total,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_hq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hq // G
    RCP_LN2: tl.constexpr = 1.4426950216

    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)
    Tq = eos_q - bos_q
    Tk = eos_k - bos_k

    p_q = tl.make_block_ptr(q + (bos_q * HQ + i_hq) * K, (Tq, K), (HQ * K, 1), (i_tl * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos_q * HQ + i_hq) * K, (Tq, K), (HQ * K, 1), (i_tl * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos_q * HQ + i_hq) * V, (Tq, V), (HQ * V, 1), (i_tl * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_tl * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_tl * BT,), (BT,), (0,))

    o_qi = i_tl * BT + tl.arange(0, BT)
    m_q = o_qi < Tq
    b_pos = tl.load(pos_ids + bos_q + o_qi, mask=m_q, other=0).to(tl.int32)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    max_pos = tl.max(tl.where(m_q, b_pos, 0))
    min_pos = tl.min(tl.where(m_q, b_pos, 2147483647))
    hi = (max_pos // BS + 1) * BS
    split = (min_pos // BS) * BS

    # KV blocks below min_pos are fully causal.
    for i_s in range(0, split, BS):
        p_k = tl.make_block_ptr(k + (bos_k * H + i_h) * K, (K, Tk), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos_k * H + i_h) * V, (V, Tk), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    # Apply the causal mask in the diagonal region.
    for i_s in range(split, hi, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < Tk
        p_k = tl.make_block_ptr(k + (bos_k * H + i_h) * K, (K, Tk), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos_k * H + i_h) * V, (V, Tk), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_s = tl.where((b_pos[:, None] >= o_k[None, :]) & m_k[None, :] & m_q[:, None], b_s, float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['Tq_total'])
def sq_full_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    pos_ids,
    q_start_blk,   # int32 [NTk] first compressed-query block to consider
    scale,
    cu_seqlens_q,
    cu_seqlens_k,
    chunk_indices,
    Tq_total,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_hq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hq // G
    RCP_LN2: tl.constexpr = 1.4426950216

    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32)
    eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
    bos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32)
    eos_k = tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)
    Tq = eos_q - bos_q
    Tk = eos_k - bos_k

    p_k = tl.make_block_ptr(k + (bos_k * H + i_h) * K, (Tk, K), (H * K, 1), (i_tl * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos_k * H + i_h) * V, (Tk, V), (H * V, 1), (i_tl * BT, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos_k * HQ + i_hq) * K, (Tk, K), (HQ * K, 1), (i_tl * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos_k * HQ + i_hq) * V, (Tk, V), (HQ * V, 1), (i_tl * BT, i_v * BV), (BT, BV), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_tl * BT + tl.arange(0, BT)
    m_kk = o_k < Tk

    # The causal boundary spans at most two compressed-query blocks.
    j0 = tl.load(q_start_blk + i_t).to(tl.int32)
    nq_blk = tl.cdiv(Tq, BT)
    j_end = tl.minimum(j0 + 2, nq_blk)

    # Apply the causal mask in the diagonal region.
    for i_s in range(j0 * BT, j_end * BT, BS):
        o_qi = i_s + tl.arange(0, BS)
        m_q = o_qi < Tq
        b_pos = tl.load(pos_ids + bos_q + o_qi, mask=m_q, other=0).to(tl.int32)

        p_q = tl.make_block_ptr(q + (bos_q * HQ + i_hq) * K, (Tq, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos_q * HQ + i_hq) * V, (Tq, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_s,), (BS,), (0,))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        b_p = tl.where((o_k[:, None] <= b_pos[None, :]) & m_q[None, :] & m_kk[:, None],
                       exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    # Remaining query blocks are fully causal.
    for i_s in range(j_end * BT, nq_blk * BT, BS):
        o_qi = i_s + tl.arange(0, BS)
        m_q = o_qi < Tq

        p_q = tl.make_block_ptr(q + (bos_q * HQ + i_hq) * K, (Tq, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos_q * HQ + i_hq) * V, (Tq, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos_q * HQ + i_hq, (Tq,), (HQ,), (i_s,), (BS,), (0,))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    b_dk *= scale
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def _fwd(q_sel, k, v, pos_ids, scale, sel: SelInfo):
    Sq, HQ, K = q_sel.shape
    Tk_total, H, V = v.shape
    G = HQ // H
    BT, BS, BK, BV, num_warps = get_fwd_config(K, V, sel.T, q_sel.device.index or 0)
    NV = triton.cdiv(V, BV)
    assert triton.cdiv(K, BK) == 1, "K must fit in a single block (<=256)"

    chunk_indices = prepare_chunk_indices(sel.cu_seqlens_q, BT)
    NT = chunk_indices.shape[0]

    o = torch.empty(Sq, HQ, V, dtype=v.dtype, device=q_sel.device)
    lse = torch.empty(Sq, HQ, dtype=torch.float, device=q_sel.device)
    if NT > 0:
        grid = (NV, NT, HQ)
        sq_full_attn_fwd_kernel[grid](
            q=q_sel, k=k, v=v, o=o, lse=lse, pos_ids=pos_ids, scale=scale,
            cu_seqlens_q=sel.cu_seqlens_q, cu_seqlens_k=sel.cu_seqlens_k,
            chunk_indices=chunk_indices, Tq_total=Sq,
            H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV,
            num_warps=num_warps,
        )
    return o, lse


def _bwd_preprocess(o_sel, do_sel):
    V = o_sel.shape[-1]
    delta = torch.empty_like(o_sel[..., 0], dtype=torch.float)
    sq_full_attn_bwd_preprocess_kernel[(delta.numel(),)](
        o=o_sel, do=do_sel, delta=delta, B=triton.next_power_of_2(V), V=V,
    )
    return delta


def _compute_q_start_blk(pos_ids, sel: SelInfo, chunk_indices_k, BT):
    """First compressed-query block (in BT units) that a kv chunk can attend to.

    For each kv chunk row, find the first compressed query with real position
    `pos >= kv_block_start`.
    """
    NTk = chunk_indices_k.shape[0]
    device = pos_ids.device
    if NTk == 0:
        return torch.empty(0, dtype=torch.int32, device=device)

    # Offset positions by sequence so searchsorted cannot cross segments.
    cu_q = sel.cu_seqlens_q.to(torch.int64)                  # [Nseq+1]
    i_n = chunk_indices_k[:, 0].to(torch.int64)              # [NTk] sequence id
    i_tl = chunk_indices_k[:, 1].to(torch.int64)             # [NTk] kv block id
    kv0 = i_tl * BT                                          # [NTk] kv block start
    bos_q = cu_q[i_n]                                        # [NTk] segment start in compressed q

    BIG = int(sel.T) + 1                                     # stride > any position
    counts = cu_q[1:] - cu_q[:-1]                            # [Nseq] queries per sequence
    seg_id = torch.repeat_interleave(
        torch.arange(sel.Nseq, device=device, dtype=torch.int64), counts
    ) if sel.Sq > 0 else torch.empty(0, device=device, dtype=torch.int64)
    keys = seg_id * BIG + pos_ids.to(torch.int64)           # [Sq] globally ascending
    query = i_n * BIG + kv0                                  # [NTk]
    glob_first = torch.searchsorted(keys, query)            # [NTk] global index
    local_first = (glob_first - bos_q).clamp_min_(0)        # within-segment index
    q_start = (local_first // BT).to(torch.int32)
    return q_start


def _bwd(q_sel, k, v, o_sel, lse, do_sel, pos_ids, scale, sel: SelInfo):
    Sq, HQ, K = q_sel.shape
    Tk_total, H, V = v.shape
    G = HQ // H
    BT, BS, BK, BV, num_warps = get_bwd_config(K, V, q_sel.device.index or 0)
    NV = triton.cdiv(V, BV)
    assert NV == 1, "V must fit in a single block for the backward kernels"

    chunk_indices_q = prepare_chunk_indices(sel.cu_seqlens_q, BT)
    chunk_indices_k = prepare_chunk_indices(sel.cu_seqlens_k, BT)
    NTq = chunk_indices_q.shape[0]
    NTk = chunk_indices_k.shape[0]

    delta = _bwd_preprocess(o_sel, do_sel)

    dq = torch.zeros(Sq, HQ, K, dtype=torch.float, device=q_sel.device)
    dk = torch.zeros(Tk_total, HQ, K, dtype=torch.float, device=q_sel.device)
    dv = torch.zeros(Tk_total, HQ, V, dtype=torch.float, device=q_sel.device)

    if NTq > 0:
        sq_full_attn_bwd_kernel_dq[(NV, NTq, HQ)](
            q=q_sel, k=k, v=v, lse=lse, delta=delta, do=do_sel, dq=dq,
            pos_ids=pos_ids, scale=scale,
            cu_seqlens_q=sel.cu_seqlens_q, cu_seqlens_k=sel.cu_seqlens_k,
            chunk_indices=chunk_indices_q, Tq_total=Sq,
            H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV,
            num_warps=num_warps,
        )
    if NTk > 0:
        q_start_blk = _compute_q_start_blk(pos_ids, sel, chunk_indices_k, BT)
        sq_full_attn_bwd_kernel_dkv[(NV, NTk, HQ)](
            q=q_sel, k=k, v=v, lse=lse, delta=delta, do=do_sel, dk=dk, dv=dv,
            pos_ids=pos_ids, q_start_blk=q_start_blk, scale=scale,
            cu_seqlens_q=sel.cu_seqlens_q, cu_seqlens_k=sel.cu_seqlens_k,
            chunk_indices=chunk_indices_k, Tq_total=Sq,
            H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV,
            num_warps=num_warps,
        )

    dk = reduce(dk, 't (h g) k -> t h k', g=G, reduction='sum')
    dv = reduce(dv, 't (h g) v -> t h v', g=G, reduction='sum')
    return dq, dk, dv


class SQFullAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, sel: SelInfo):
        B, T, HQ, K = q.shape
        _, _, H, V = v.shape
        q_flat = q.reshape(B * T, HQ, K)
        k_flat = k.reshape(B * T, H, K)
        v_flat = v.reshape(B * T, H, V)

        q_sel = q_flat.index_select(0, sel.q_idx)  # [Sq, HQ, K]
        o_sel, lse = _fwd(q_sel, k_flat, v_flat, sel.pos_ids, scale, sel)

        ctx.save_for_backward(q_sel, k_flat, v_flat, o_sel, lse, sel.q_idx, sel.pos_ids)
        ctx.sel = sel
        ctx.scale = scale
        ctx.shapes = (B, T, HQ, H, K, V)

        o = q.new_zeros(B * T, HQ, V)
        o.index_copy_(0, sel.q_idx, o_sel.to(o.dtype))
        return o.view(B, T, HQ, V)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q_sel, k_flat, v_flat, o_sel, lse, q_idx, pos_ids = ctx.saved_tensors
        sel = ctx.sel
        scale = ctx.scale
        B, T, HQ, H, K, V = ctx.shapes

        do_flat = do.reshape(B * T, HQ, V)
        do_sel = do_flat.index_select(0, q_idx)

        dq_sel, dk, dv = _bwd(q_sel, k_flat, v_flat, o_sel, lse, do_sel, pos_ids, scale, sel)

        dq = q_sel.new_zeros(B * T, HQ, K)
        dq.index_copy_(0, q_idx, dq_sel.to(dq.dtype))
        dq = dq.view(B, T, HQ, K)
        dk = dk.view(B, T, H, K).to(k_flat.dtype)
        dv = dv.view(B, T, H, V).to(v_flat.dtype)
        return dq, dk, dv, None, None


def sq_full_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selection: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Selected-query full causal attention.

    Args:
        q: queries `[1, total_tokens, HQ, K]`.
        k: keys    `[1, total_tokens, H, K]` (GQA when HQ % H == 0).
        v: values  `[1, total_tokens, H, V]`.
        selection: per-token bool mask `[1, total_tokens]`.
        scale: attention scale; defaults to `K ** -0.5`.
        cu_seqlens: required `[Nseq+1]` int tensor for packed varlen input.

    Returns:
        o: `[1, total_tokens, HQ, V]`; rows for unselected query tokens are 0.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be [B, T, H, D]"
    assert HQ % H == 0, "HQ must be a multiple of H (GQA)"
    assert selection.dtype == torch.bool and selection.shape == (B, T), "selection must be a bool mask [B, T]"
    assert B == 1, "sq_full_attn only supports B=1 packed-varlen input"
    assert cu_seqlens is not None, "sq_full_attn requires explicit cu_seqlens"
    if scale is None:
        scale = K ** -0.5

    sel = build_selection(selection, B, T, cu_seqlens)
    return SQFullAttentionFunction.apply(q, k, v, scale, sel)
