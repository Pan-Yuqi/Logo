# -*- coding: utf-8 -*-
"""LoGo: Token-Level Dynamic Local-Global Attention."""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)

from logo.cache import Cache
from logo.configuration_logo import LoGoConfig
from logo.layers import Attention, LoGoAttention
from logo.modeling_logo import (
    LoGoForCausalLM,
    LoGoModel,
    LoGoPreTrainedModel,
)
from logo.update import update_gate_thres

AutoConfig.register(LoGoConfig.model_type, LoGoConfig)
AutoModel.register(LoGoConfig, LoGoModel)
AutoModelForCausalLM.register(LoGoConfig, LoGoForCausalLM)

__version__ = "0.1.0"

__all__ = [
    "Cache",
    "LoGoConfig",
    "Attention",
    "LoGoAttention",
    "LoGoModel",
    "LoGoPreTrainedModel",
    "LoGoForCausalLM",
    "update_gate_thres",
]
