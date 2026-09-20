# -*- coding: utf-8 -*-
"""Training-time utilities for LoGo's global-attention budget."""

import os
from typing import Optional

import torch
import torch.distributed as dist

__all__ = ["update_gate_thres"]


GATE_THRES_UPDATE_RATE = float(os.getenv("GATE_THRES_UPDATE_RATE", "0.0005"))
TARGET_GLOBAL_RATIO = float(os.getenv("TARGET_GLOBAL_RATIO", "0.5"))


def update_gate_thres(
    model,
    target_global_ratio: float = TARGET_GLOBAL_RATIO,
    update_rate: float = GATE_THRES_UPDATE_RATE,
    process_group=None,
) -> Optional[torch.Tensor]:
    """Update gate thresholds from the accumulated global-token ratios."""
    if not 0.0 <= target_global_ratio <= 1.0:
        raise ValueError(f"`target_global_ratio` must be in [0, 1], got {target_global_ratio}")
    if update_rate < 0.0:
        raise ValueError(f"`update_rate` must be non-negative, got {update_rate}")

    gate_thres_list = []
    gate_avg_list = []
    global_token_count_list = []
    total_token_count_list = []

    for module in model.modules():
        if all(
            hasattr(module, name)
            for name in ("gate_thres", "gate_avg", "global_token_count", "total_token_count")
        ):
            gate_thres_list.append(module.gate_thres)
            gate_avg_list.append(module.gate_avg)
            global_token_count_list.append(module.global_token_count)
            total_token_count_list.append(module.total_token_count)

    if not gate_thres_list:
        return None

    gate_thres = torch.stack(gate_thres_list, dim=0)
    global_token_count = torch.stack(global_token_count_list, dim=0)
    total_token_count = torch.stack(total_token_count_list, dim=0)

    with torch.no_grad():
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(total_token_count, op=dist.ReduceOp.SUM, group=process_group)

        has_observations = total_token_count > 0
        global_token_ratio = torch.where(
            has_observations,
            global_token_count / total_token_count.clamp_min(1),
            torch.zeros_like(global_token_count),
        )
        update = update_rate * torch.sign(global_token_ratio - target_global_ratio)
        gate_thres.add_(torch.where(has_observations, update, torch.zeros_like(update)))

        for layer_gate_thres, updated_gate_thres in zip(gate_thres_list, gate_thres):
            layer_gate_thres.copy_(updated_gate_thres)

        for gate_avg, global_count, total_count in zip(
            gate_avg_list, global_token_count_list, total_token_count_list
        ):
            gate_avg.zero_()
            global_count.zero_()
            total_count.zero_()

    return global_token_ratio
