"""Inspection utilities for masked temporal transformer: attention and latents."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from behavex.models.transformer import MaskedTemporalTransformer


def _encoder_forward_with_attention(
    encoder: nn.TransformerEncoder,
    src: torch.Tensor,
    *,
    average_attn_weights: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run encoder layer-by-layer, calling self_attn with need_weights=True.

    Matches the standard TransformerEncoder forward (no encoder-level norm).
    Returns (output, list of attention tensors per layer).
    Each attention tensor: (B, T, T) if average_attn_weights else (B, nhead, T, T).
    """
    attention_list: list[torch.Tensor] = []
    output = src
    for layer in encoder.layers:
        # Self-attention with weights
        attn_out, attn_weights = layer.self_attn(
            output,
            output,
            output,
            need_weights=True,
            average_attn_weights=average_attn_weights,
        )
        if attn_weights is not None:
            attention_list.append(attn_weights.detach())
        output = output + layer.dropout1(attn_out)
        output = layer.norm1(output)
        output = output + layer.dropout2(
            layer.linear2(layer.activation(layer.linear1(output)))
        )
        output = layer.norm2(output)
    if encoder.norm is not None:
        output = encoder.norm(output)
    return output, attention_list


def forward_with_attention(
    model: "MaskedTemporalTransformer",
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    average_attn_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Full forward pass returning recon, latent, mask, and per-layer attention.

    Uses the same embedding + mask + RoPE + norm as the model, then runs the
    encoder with need_weights=True per layer. Output latent matches model(x)[1].

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
    emb = model.pos_encoder(emb)
    emb = model.norm(emb)

    if model.return_layer_mean:
        # Same as in transformer: manual layer loop, then mean
        layer_outs: list[torch.Tensor] = []
        attn_all: list[torch.Tensor] = []
        h = emb
        for layer in model.encoder.layers:
            attn_out, attn_weights = layer.self_attn(
                h, h, h,
                need_weights=True,
                average_attn_weights=average_attn_weights,
            )
            if attn_weights is not None:
                attn_all.append(attn_weights.detach())
            h = h + layer.dropout1(attn_out)
            h = layer.norm1(h)
            h = h + layer.dropout2(
                layer.linear2(layer.activation(layer.linear1(h)))
            )
            h = layer.norm2(h)
            layer_outs.append(h)
        latent = torch.stack(layer_outs, dim=0).mean(dim=0)
        if model.encoder.norm is not None:
            latent = model.encoder.norm(latent)
        attention_per_layer = attn_all
    else:
        latent, attention_per_layer = _encoder_forward_with_attention(
            model.encoder, emb, average_attn_weights=average_attn_weights
        )

    recon = model.reconstruction_head(latent)
    return recon, latent, mask, attention_per_layer
