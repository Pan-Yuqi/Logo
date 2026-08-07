# -*- coding: utf-8 -*-
"""Common utilities for the LoGo selected-query full-attention operator.

This module provides:
  - numerical error helpers (get_abs_err / get_err_ratio / assert_close)
  - selection metadata construction (build_selection)
  - block-size configs chosen per GPU shared-memory tier

All metadata is laid out so the optimized "selected-query" kernel and the
"dense reference" kernel (full selection) can share the exact same Triton
kernels (see selected_full_attn.py / dense_full_attn.py).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton

from .triton_utils import check_shared_mem

__all__ = [
    "SelInfo",
    "build_selection",
    "prepare_chunk_indices",
    "get_fwd_config",
    "get_bwd_config",
    "get_abs_err",
    "get_err_ratio",
    "assert_close",
]


# ----------------------------------------------------------------------------
# numerical error helpers
# ----------------------------------------------------------------------------
def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6e} ratio: {get_err_ratio(ref, tri):.6e}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return msg
    assert error_rate < ratio, msg
    return msg


# ----------------------------------------------------------------------------
# selection metadata
# ----------------------------------------------------------------------------
@dataclass
class SelInfo:
    """Metadata describing which query tokens are selected for full attention.

    Layouts follow a unified "varlen-style" compressed-query axis: all selected
    queries across all sequences are packed into a single axis of length Sq, in
    ascending (sequence, position) order.

    Fields:
        q_idx        : int64  [Sq]      flat indices into the (B*T) query rows
        pos_ids      : int32  [Sq]      real position (within its sequence) of each
                                        compressed query (0-based)
        cu_seqlens_q : int32  [Nseq+1]  prefix-sum of #selected per sequence
                                        (compressed-query blocking)
        cu_seqlens_k : int32  [Nseq+1]  kv extent per sequence (original lengths)
        B, T         : int              original batch / seqlen (fixed-batch);
                                        for varlen B==1 and T == total tokens
        Nseq         : int              number of sequences
        Sq           : int              number of selected queries
    """

    q_idx: torch.Tensor
    pos_ids: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    B: int
    T: int
    Nseq: int
    Sq: int


def build_selection(
    selection: torch.Tensor,
    B: int,
    T: int,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> SelInfo:
    """Build SelInfo from a boolean selection mask.

    The kernels support the varlen-style representation used by the production
    path: `B == 1`, `T == total_tokens`, and an explicit `cu_seqlens` that
    describes the logical sequences packed into that one row.

    Args:
        selection: bool mask `[1, total_tokens]`.
        B, T:      must be `B=1`, `T=total_tokens`.
        cu_seqlens: required `[Nseq+1]` int tensor for packed varlen input.
    """
    device = selection.device
    flat = selection.reshape(-1)                                    # [B*T] bool

    cu_seqlens_k = cu_seqlens.to(device=device, dtype=torch.int32)
    Nseq = cu_seqlens_k.numel() - 1

    # global nonzero -> indices are ascending, i.e. sorted by (sequence, position)
    q_idx = flat.nonzero(as_tuple=False).squeeze(-1)                # [Sq] int64
    cu_k64 = cu_seqlens_k.to(torch.int64)
    # sequence id of each selected query, then its in-sequence position
    seg_id = torch.searchsorted(cu_k64, q_idx, right=True) - 1      # [Sq]
    pos_ids = (q_idx - cu_k64[seg_id]).to(torch.int32)             # [Sq]
    counts = torch.bincount(seg_id, minlength=Nseq)                # [Nseq]
    cu_seqlens_q = torch.zeros(Nseq + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = counts.cumsum(0).to(torch.int32)

    return SelInfo(
        q_idx=q_idx,
        pos_ids=pos_ids,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        B=B,
        T=T,
        Nseq=Nseq,
        Sq=int(q_idx.numel()),
    )


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Map a flat block id to (sequence_id, local_block_id).

    Returns int32 `[NT, 2]` where NT = sum_s ceil(len_s / chunk_size). Robust to
    zero-length (fully-unselected) sequences.
    """
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    rows = []
    for sid, n in enumerate(lens):
        nb = triton.cdiv(int(n), chunk_size)
        for b in range(nb):
            rows.append((sid, b))
    if len(rows) == 0:
        return torch.empty(0, 2, dtype=torch.int32, device=cu_seqlens.device)
    return torch.tensor(rows, dtype=torch.int32, device=cu_seqlens.device)


def get_fwd_config(K: int, V: int, T: int, device_index: int = 0):
    """Block sizes / num_warps for the forward kernel."""
    BT = 128
    if check_shared_mem("hopper", device_index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
        num_warps = 8
    elif check_shared_mem("ampere", device_index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
        num_warps = 4
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
        num_warps = 2
    return BT, BS, BK, BV, num_warps


def get_bwd_config(K: int, V: int, device_index: int = 0):
    """Block sizes / num_warps for the backward kernels."""
    if check_shared_mem("hopper", device_index):
        BT = 128
        BS = 64
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        num_warps = 8
    elif check_shared_mem("ampere", device_index):
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        BT = 128 if K <= 64 else 64
        num_warps = 4
    else:
        BT = 64
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = min(max(triton.next_power_of_2(V), 16), 64)
        num_warps = 2
    return BT, BS, BK, BV, num_warps
