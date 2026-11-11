"""ESM C 300M as the epitope embedder of the img1 heads.

The released checkpoint (esmc_300m_2024_12_v0_fp16.pth, 308 tensors) uses the original EvolutionaryScale key layout. Current `esm` keeps ESM C in a Hugging Face style
layout (fused projections under other names), so the keys are renamed at load: renames only, every tensor is passed through unchanged. Only the encoder is built;
the language-model head (`sequence_head.*`) is not needed to embed and is dropped. Attention runs through SDPA; flash-attn is not used and not needed.
"""
import re

import torch
from torch import nn
from esm.models.esmc import EsmcConfig, EsmcModel

D_MODEL, N_HEADS, N_LAYERS = 960, 15, 30

_BLOCK = {
    "attn.layernorm_qkv.0.weight": "attn.layernorm_qkv.layer_norm_weight",
    "attn.layernorm_qkv.0.bias": "attn.layernorm_qkv.layer_norm_bias",
    "attn.layernorm_qkv.1.weight": "attn.layernorm_qkv.weight",
    "ffn.0.weight": "ffn.layer_norm_weight",
    "ffn.0.bias": "ffn.layer_norm_bias",
    "ffn.1.weight": "ffn.fc1_weight",
    "ffn.3.weight": "ffn.fc2_weight",
}


def to_encoder_layout(state):
    """Original ESM C key layout -> the layout of `EsmcModel` (the encoder without the language-model head)."""
    out = {}
    for key, value in state.items():
        block = re.match(r"transformer\.blocks\.(\d+)\.(.*)", key)
        if block:
            n, rest = block.groups()
            out[f"transformer.blocks.{n}.{_BLOCK.get(rest, rest)}"] = value
        elif key in ("transformer.norm.weight", "embed.weight"):
            out[key] = value
        elif key.startswith("sequence_head."):
            continue
        else:
            raise KeyError(f"unexpected ESM C key: {key}")
    return out


class Embedder(nn.Module):
    """Epitope tokens (B, L) -> last hidden state after the final norm (B, L, 960), the `embeddings` of the original ESM C output."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, tokens):
        return self.encoder(input_ids=tokens, return_dict=True).last_hidden_state


def load_embedder(path, device, dtype=torch.float16):
    config = EsmcConfig(hidden_size=D_MODEL, num_attention_heads=N_HEADS, num_hidden_layers=N_LAYERS, attn_implementation="sdpa")
    encoder = EsmcModel(config)
    encoder.load_state_dict(to_encoder_layout(torch.load(path, map_location="cpu", weights_only=True)), strict=True)
    return Embedder(encoder).to(device, dtype=dtype).eval()
