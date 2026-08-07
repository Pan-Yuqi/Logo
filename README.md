# LoGo: Token-Level Dynamic Local-Global Attention

LoGo is a token-level dynamic local-global attention mechanism that uses
**attention span as a direct proxy for attention budget allocation**. As context
lengths scale, attention becomes a primary computational bottleneck: standard
Transformers spend the same attention budget on every token regardless of its
contextual demand. LoGo instead lets each token decide how much context it needs.

Each LoGo layer contains **coupled local and global branches**:

- **Local branch** — *every* token receives efficient sliding-window (SWA)
  attention over a restricted context window.
- **Global branch** — a learned per-token scalar **gate** activates full-context
  attention only for the tokens that require long-range information.

The two branches are combined per token:

```
gate = sigmoid(gate_proj(hidden_states))
out  = (1 - gate) * local_out + gate * global_out
```

A threshold-based budget controller maintains a target global ratio without any
auxiliary loss, and **query-sparse Triton kernels** turn the reduced
global-attention computation into practical speedups. Because span allocation is
shared per layer, LoGo does not introduce head-level imbalance and remains
compatible with efficient tensor / sequence parallelism.

## Repository layout

```
LoGo/
├── logo/
│   ├── __init__.py                 # package exports + Auto* registration
│   ├── configuration_logo.py       # LoGoConfig
│   ├── modeling_logo.py            # LoGoModel / LoGoForCausalLM
│   ├── cache.py                    # dual-branch (local + global) KV cache
│   ├── modules/                    # reusable building blocks
│   │   ├── layernorm.py            #   RMSNorm
│   │   ├── rotary.py               #   RotaryEmbedding + apply_rotary_pos_emb
│   │   └── mlp.py                  #   SwiGLU MLP
│   ├── layers/                     # attention layers
│   │   ├── attn.py                 #   standard full flash-attention
│   │   ├── logo.py                 #   LoGoAttention (local + gated global)
│   │   └── utils.py                #   flash-attention forward dispatch
│   └── ops/                        # self-contained Triton operators
│       ├── selected_full_attn.py   #   sq_full_attn: query-sparse full attention
│       ├── dense_full_attn.py      #   dense references for testing
│       ├── common.py               #   selection / launch helpers
│       └── triton_utils.py         #   inlined Triton helpers
├── configs/
│   └── config_1b5.json             # 1.5B reference configuration
├── tests/
│   └── test_model.py               # import / build / forward smoke tests
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

```bash
cd LoGo
pip install -e .
# Flash-attention is required for the attention layers at runtime:
pip install flash-attn --no-build-isolation
```

Requirements: `torch>=2.1`, `transformers>=4.44,<4.52`, `triton>=3.0`,
`einops>=0.7`, and `flash-attn>=2.1`.

## Quickstart

```python
import torch
from logo import LoGoConfig, LoGoForCausalLM

config = LoGoConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,
    max_position_embeddings=8192,
    window_size=128,                     # local (SWA) window
    global_qk_param="linear_proj",       # global-branch q/k/v re-parameterization
    sparse_full_attn_backend="flash",    # "flash" (dense) or "triton" (query-sparse)
    gate_thres_init=0.5,                 # token-level span budget threshold
)

model = LoGoForCausalLM(config).cuda().to(torch.bfloat16)
input_ids = torch.randint(0, config.vocab_size, (1, 512), device="cuda")
out = model(input_ids)
print(out.logits.shape)  # [1, 512, vocab_size]
```

Loading the reference config:

```python
from logo import LoGoConfig, LoGoForCausalLM

config = LoGoConfig.from_pretrained("configs/config_1b5.json")
model = LoGoForCausalLM(config)
```

Because the model registers itself with the `transformers` Auto classes on
import, `AutoConfig`/`AutoModelForCausalLM` also work once `import logo` has run.

## Key configuration options

| Field | Default | Description |
|-------|---------|-------------|
| `window_size` | `128` | Sliding-window size for the local branch. |
| `attn_type_list` | all `0` | Per-layer type: `0` = LoGo attention, else standard attention. |
| `global_qk_param` | `"scale_offset"` | Global-branch q/k/v transform: `scale_offset` (per-channel affine) or `linear_proj` (per-head mixing matrix). |
| `sparse_full_attn_backend` | `"flash"` | Global branch backend: `flash` (dense) or `triton` (query-sparse). |
| `gate_thres_init` | `0.5` | Initial gate threshold for the token-level span budget. |
| `use_context_norm` | `True` | Per-branch RMSNorm on the attention context before gating. |

The `triton` backend (`sparse_full_attn_backend="triton"`) runs
`logo.ops.sq_full_attn`, a selected-query full-attention kernel that only computes
the rows chosen by the gate — this is where the reduced global-attention budget
becomes a real speedup.

## Testing

```bash
cd ./LoGo
python tests/test_model.py          # GPU smoke test: build + train/packed/generate
```

The smoke test builds the reference model from `configs/config_1b5.json` and
requires a CUDA GPU (flash-attention and the Triton kernels are GPU-only).

## License

Apache License 2.0.
