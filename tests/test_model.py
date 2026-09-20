# -*- coding: utf-8 -*-
"""Basic test for the LoGo model.

Builds the reference model from ``configs/config_1b5.json`` and exercises the
training forward (dense + packed ``cu_seqlens``) and the inference ``generate``
path.

Run from anywhere::

    python tests/test_model.py
"""

import os

import torch

from logo import LoGoConfig, LoGoForCausalLM

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config_1b5.json")


def main():
    config = LoGoConfig.from_pretrained(CONFIG_PATH)
    model = LoGoForCausalLM(config)
    print(model)
    print(f"num_parameters: {model.num_parameters():,}")
    model = model.cuda().to(torch.bfloat16)

    vocab_size = config.vocab_size

    print("=== Training ===")
    model.train()
    input_ids = torch.randint(0, vocab_size, (1, 8192)).cuda()
    outputs = model(input_ids, labels=input_ids.clone(), use_cache=False)
    print(outputs.loss)

    del outputs, input_ids
    torch.cuda.empty_cache()

    # packed variable-length sequences via cu_seqlens
    input_ids = torch.randint(0, vocab_size, (1, 8192)).cuda()
    cu_seqlens = torch.tensor([0, 1024, 4000, 8192], dtype=torch.int32, device=input_ids.device)
    outputs = model(input_ids, labels=input_ids.clone(), cu_seqlens=cu_seqlens, use_cache=False)
    print(outputs.loss)

    del outputs, input_ids, cu_seqlens
    torch.cuda.empty_cache()

    print("=== Inference ===")
    model.eval()
    input_ids = torch.randint(0, vocab_size, (2, 20)).cuda()
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :5] = 0
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        do_sample=False,
        max_new_tokens=64,
    )
    print(output.shape)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This test requires a CUDA GPU (flash-attn + Triton kernels).")
    main()
