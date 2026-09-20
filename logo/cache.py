# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers

__all__ = ["Cache"]


class Cache(transformers.cache_utils.Cache):
    """Key/value cache for LoGo's dual-branch attention.

    Each layer stores a dictionary of states. LoGo layers keep two attention
    states side by side:

    * ``attn_state``      -- key/value for the local (sliding-window) branch,
    * ``attn_state_full`` -- key/value for the global (full) branch.

    The local branch retains a rolling ``window_size`` buffer, while the global
    branch keeps the full sequence.
    """

    is_compileable = True

    def __init__(self, seen_tokens: int = 0) -> "Cache":
        super().__init__()
        self.states: List[Dict[str, Any]] = []
        # Used in `generate` to keep tally of how many tokens the cache has seen.
        self._seen_tokens = seen_tokens

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        attn_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_state_full: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update the cache for ``layer_idx`` and return the updated state dict.

        Args:
            attn_state: New key/value states for the local (SWA) branch.
            attn_state_full: New key/value states for the global (full) branch.
            layer_idx: Index of the layer being updated.
            offset: Number of new tokens being processed.
            cache_kwargs: Extra arguments; ``window_size`` bounds the local buffer.
        """
        cache_kwargs = cache_kwargs or {}

        # Update the number of seen tokens (once per step, on the first layer).
        if layer_idx == 0:
            self._seen_tokens += offset

        window_size = None
        if attn_state is not None:
            if not isinstance(attn_state, tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
            window_size = cache_kwargs.get("window_size", None)
            if window_size is not None and window_size <= 0:
                raise ValueError(f"`window_size` must be positive, got {window_size}")

        attn_state_for_attention = attn_state

        if len(self.states) <= layer_idx:
            # first write for this layer
            attn_state_to_cache = attn_state
            if (
                attn_state is not None
                and window_size is not None
                and attn_state[0].shape[-2] > window_size
            ):
                attn_state_to_cache = (
                    attn_state[0][..., -window_size:, :].contiguous(),
                    attn_state[1][..., -window_size:, :].contiguous(),
                )
            state = dict(attn_state=attn_state_to_cache, attn_state_full=attn_state_full)
            self.states.append(state)
        else:
            state = self.states[layer_idx]
            if attn_state is not None:
                key_state, value_state = state["attn_state"]
                attn_state_for_attention = (
                    torch.cat([key_state, attn_state[0]], dim=-2),
                    torch.cat([value_state, attn_state[1]], dim=-2),
                )
                attn_state_to_cache = attn_state_for_attention
                if (
                    window_size is not None
                    and attn_state_for_attention[0].shape[-2] > window_size
                ):
                    attn_state_to_cache = (
                        attn_state_for_attention[0][..., -window_size:, :].contiguous(),
                        attn_state_for_attention[1][..., -window_size:, :].contiguous(),
                    )
                state["attn_state"] = attn_state_to_cache
            if attn_state_full is not None:
                if state["attn_state_full"] is None:
                    state["attn_state_full"] = attn_state_full
                else:
                    state["attn_state_full"] = (
                        torch.cat([state["attn_state_full"][0], attn_state_full[0]], dim=-2),
                        torch.cat([state["attn_state_full"][1], attn_state_full[1]], dim=-2),
                    )

        if attn_state_for_attention is None:
            attn_state_for_attention = state["attn_state"]
        output_state = state.copy()
        output_state["attn_state"] = attn_state_for_attention
        return output_state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the number of cached tokens (0 if this layer has no state yet)."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """The cache has no fixed maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple] = None, seen_tokens: int = 0) -> "Cache":
        """Build a ``Cache`` from the legacy tuple-of-states format."""
        cache = cls(seen_tokens)
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])

            if seen_tokens == 0:
                for state in cache.states:
                    full_state = state.get("attn_state_full")
                    local_state = state.get("attn_state")
                    inferred_state = full_state if full_state is not None else local_state
                    if inferred_state is not None:
                        cache._seen_tokens = inferred_state[0].shape[-2]
                        break
        return cache
