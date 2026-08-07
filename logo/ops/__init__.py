# -*- coding: utf-8 -*-
"""LoGo operators: selected-query full causal attention.

Only the query tokens selected by the per-token gate participate in the full
causal attention; unselected query rows produce zero output / zero gradient,
saving compute proportionally to the masked fraction. This is the query-sparse
global branch that turns LoGo's reduced global-attention budget into a practical
speedup.
"""

from .common import (
    SelInfo,
    assert_close,
    build_selection,
    get_abs_err,
    get_err_ratio,
)
from .dense_full_attn import sq_full_attn_dense_ref, sdpa_causal_ref
from .selected_full_attn import sq_full_attn

__all__ = [
    "sq_full_attn",
    "sq_full_attn_dense_ref",
    "sdpa_causal_ref",
    "build_selection",
    "SelInfo",
    "get_abs_err",
    "get_err_ratio",
    "assert_close",
]
