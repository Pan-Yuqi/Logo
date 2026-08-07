# -*- coding: utf-8 -*-
"""Small, self-contained Triton / autograd helpers for the LoGo operators.

These utilities are lightweight ports of a handful of helpers that the selected
full-attention kernels rely on, kept local so the ``logo.ops`` package has no
external dependency beyond ``torch`` and ``triton``.
"""

import functools
from enum import Enum

import torch
import triton
import triton.language as tl

__all__ = [
    "exp2",
    "log2",
    "contiguous",
    "autocast_custom_fwd",
    "autocast_custom_bwd",
    "check_shared_mem",
]

# ----------------------------------------------------------------------------
# Triton math intrinsics
# ----------------------------------------------------------------------------
exp2 = tl.math.exp2
log2 = tl.log2


# ----------------------------------------------------------------------------
# autograd helpers
# ----------------------------------------------------------------------------
def contiguous(fn):
    """Make every tensor argument contiguous before calling ``fn``.

    Triton block pointers assume contiguous inner strides, so autograd
    ``forward``/``backward`` methods are wrapped with this decorator.
    """

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
        )

    return wrapper


# `torch.amp.custom_fwd/custom_bwd` require a device_type since torch>=2.4.
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


# ----------------------------------------------------------------------------
# shared-memory capacity probe (used to pick block sizes per GPU arch)
# ----------------------------------------------------------------------------
class _Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # conservative default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def _get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
            for i in range(torch.cuda.device_count())
        ]
    except BaseException:
        return [-1]


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """Return True if device ``tensor_idx`` has enough shared memory for ``arch``."""
    try:
        max_shared_memory = _get_all_max_shared_mem()[tensor_idx]
        return max_shared_memory >= _Backend.get_shared_memory(arch)
    except Exception:
        return False
