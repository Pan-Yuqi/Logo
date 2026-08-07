# -*- coding: utf-8 -*-
# Adapted from HuggingFace transformers' flash-attention utilities.

"""Flash-attention forward helpers shared by the LoGo attention layers.

This is a trimmed port of the flash-attention wrapper used inside
``transformers``.  It supports three common code paths:

* explicit ``cu_seqlens`` for pre-packed variable-length batches,
* a 2D padding ``attention_mask`` (padded batches), and
* packed sequences described by ``position_ids``.
"""

import inspect
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
else:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _flash_supports_window_size = None

__all__ = ["flash_attention_forward"]


def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Retrieve the indexing data required to unpad ragged tensors."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    """Unpad query/key/value tensors into a single ragged dimension."""
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa2_from_position_ids(query, key, value, position_ids, max_seqlen):
    """Derive ``flash_attn_varlen_func`` arguments from packed ``position_ids``."""
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    if max_seqlen is not None:
        max_length = max_seqlen
    else:
        max_length = position_ids.max() + 1

    return query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length)


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    max_seqlen: int = None,
    **kwargs,
):
    """Dispatch to the appropriate flash-attention kernel.

    Tensors are laid out as ``[batch, seq_len, num_heads, head_dim]``.  When
    ``sliding_window`` is set the SWA (local) window is applied; passing ``None``
    yields full (global) attention.
    """
    if flash_attn_func is None:
        raise ImportError("flash-attn is required for LoGo attention. Install `flash-attn>=2.1`.")

    if not use_top_left_mask:
        causal = is_causal
    else:
        # flash_attn<2.1 uses a top-left aligned causal mask; skip masking for the q_len==1 decode step.
        causal = is_causal and query_length != 1

    # A 2D padding mask is only meaningful when it actually contains padding.
    attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

    # key_states.shape[1] is the source (key/value) sequence length.
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens_k = cu_seqlens
        max_seqlen_in_batch_q = max_seqlen_in_batch_k = (
            cu_seqlens.diff().max().item() if max_seqlen is None else max_seqlen
        )
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        # varlen output is [total_tokens, num_heads, head_dim]; the caller reshapes to (bsz, q_len, -1).

    # Contains at least one padding token in the sequence.
    elif attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # Packed sequences described by position_ids (pre-fill / training only).
    elif position_ids is not None and not (torch.diff(position_ids, dim=-1) >= 0).all() and query_length != 1:
        batch_size = query_states.size(0)
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids, max_seqlen
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return attn_output
