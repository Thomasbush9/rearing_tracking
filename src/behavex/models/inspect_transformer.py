"""Inspection utilities for masked temporal transformer: attention and latents."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from behavex.models.transformer import MaskedTemporalTransformer


def forward_with_attention(
    model: "MaskedTemporalTransformer",
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    average_attn_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Full forward pass returning recon, latent, mask, and per-layer attention.

    Replicates the model's embedding + mask + encoder logic, but runs each
    RoPETransformerEncoderLayer with need_weights=True to capture attention maps.

    Args:
        model: MaskedTemporalTransformer or MaskedPredictiveTemporalTransformer.
        x: (B, T, F) input windows.
        mask: (B, T) bool, True = masked. If None, no masking (all False).
        average_attn_weights: If True, attention shape (B, T, T); else (B, nhead, T, T).

    Returns:
        recon: (B, T, F)
        latent: (B, T, d_model)
        mask: (B, T) bool
        attention_per_layer: list of length num_layers, each (B, T, T) or (B, nhead, T, T).
    """
    B, T, F = x.shape
    if mask is None:
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

    emb = model.embedding(x)
    emb = emb + (mask.unsqueeze(-1) * (model.mask_token - emb))

    attention_per_layer: list[torch.Tensor] = []

    if model.return_layer_mean:
        layer_outs: list[torch.Tensor] = []
        h = emb
        for layer in model.layers:
            h, attn_weights = layer(h, is_causal=model.causal, need_weights=True)
            if attn_weights is not None:
                if average_attn_weights:
                    attn_weights = attn_weights.mean(dim=1)  # (B, H, T, T) -> (B, T, T)
                attention_per_layer.append(attn_weights)
            layer_outs.append(h)
        latent = model.norm(torch.stack(layer_outs, dim=0).mean(dim=0))
    else:
        h = emb
        for layer in model.layers:
            h, attn_weights = layer(h, is_causal=model.causal, need_weights=True)
            if attn_weights is not None:
                if average_attn_weights:
                    attn_weights = attn_weights.mean(dim=1)
                attention_per_layer.append(attn_weights)
        latent = model.norm(h)

    recon = model.reconstruction_head(latent)
    return recon, latent, mask, attention_per_layer
