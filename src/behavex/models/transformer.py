import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model:int):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.Tensor):
        output = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6) * self.weight
        return output

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d:int, base:int=10_000):

        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None
    def build_cache(self, x: torch.Tensor):
        seq_len = x.shape[1]  # time_window dimension
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)  # (d/2,)
        seq_idx = torch.arange(seq_len, device=x.device).float()  # (time_window,)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  # (time_window, d/2)
        idx_theta_2 = torch.cat([idx_theta, idx_theta], dim=1)  # (time_window, d)

        self.cos_cached = idx_theta_2.cos()[:, None, :]  # (time_window, 1, d)
        self.sin_cached = idx_theta_2.sin()[:, None, :]  # (time_window, 1, d)

    def forward(self, x: torch.Tensor):
        # x: (Batch, time_window, feature_dim)
        self.build_cache(x)
        neg_half = self._neg_half(x)
        cos = self.cos_cached[:x.shape[1]].permute(1, 0, 2)  # (1, T, D) for (B,T,D) broadcast
        sin = self.sin_cached[:x.shape[1]].permute(1, 0, 2)
        x_rope = (x * cos) + (neg_half * sin)
        return x_rope
    def _neg_half(self, x:torch.Tensor):
        d_2 = self.d //2
        return torch.cat([-x[..., d_2:], x[...,  :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings applied to Q and K only."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.rope = RotaryPositionalEmbeddings(self.head_dim)

    def forward(
        self, x: torch.Tensor, is_causal: bool = False, need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        Q, K, V = qkv.unbind(0)  # each (B, H, T, Dh)

        # Apply RoPE to Q and K per head
        Q = self._apply_rope(Q)
        K = self._apply_rope(K)

        if need_weights:
            # Manual attention to capture weights (for inspection)
            scale = self.head_dim ** -0.5
            attn_scores = (Q @ K.transpose(-2, -1)) * scale  # (B, H, T, T)
            if is_causal:
                causal = torch.triu(
                    torch.full((T, T), float("-inf"), device=x.device, dtype=attn_scores.dtype),
                    diagonal=1,
                )
                attn_scores = attn_scores + causal
            attn_weights = F.softmax(attn_scores, dim=-1)
            drop_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
            out = drop_weights @ V
            out = out.transpose(1, 2).contiguous().reshape(B, T, D)
            return self.out_proj(out), attn_weights.detach()

        # Fast path: scaled_dot_product_attention (Flash Attention when available)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=is_causal,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(out), None

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE. x: (B, H, T, Dh) -> merge B,H for RoPE -> reshape back."""
        B, H, T, Dh = x.shape
        return self.rope(x.reshape(B * H, T, Dh)).reshape(B, H, T, Dh)


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with RoPE applied inside attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, is_causal: bool = False, need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed, is_causal=is_causal, need_weights=need_weights)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if need_weights:
            return x, attn_weights
        return x


def create_timestep_mask(
    batch_size: int, seq_len: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Random mask over timesteps. Returns bool mask True = masked (predict)."""
    n_mask = max(1, int(seq_len * mask_ratio))
    noise = torch.rand(batch_size, seq_len, device=device)
    _, indices = noise.topk(n_mask, dim=1)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask


def create_value_mask(
    batch_size: int,
    seq_len: int,
    num_features: int,
    value_mask_ratio: float,
    device: torch.device,
    exclude_timestep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, T, F) bool, True = value masked. At each (b,t) where not excluded, mask ~value_mask_ratio of features.
    When exclude_timestep_mask (B, T) is given, only (b,t) with exclude_timestep_mask[b,t]==False get value masking."""
    n_value = max(1, int(num_features * value_mask_ratio))
    noise = torch.rand(batch_size, seq_len, num_features, device=device)
    _, indices = noise.topk(n_value, dim=2)
    value_mask = torch.zeros(batch_size, seq_len, num_features, dtype=torch.bool, device=device)
    value_mask.scatter_(2, indices, True)
    if exclude_timestep_mask is not None:
        value_mask[exclude_timestep_mask] = False
    return value_mask


class TemporalTransformer(nn.Module):

    def __init__(self, f_in:int, d_model:int, nhead:int, num_layers:int, dropout:float=0.1, output_size:int=1):
        super().__init__()
        self.f_in = f_in
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        dim_feedforward = 2 * d_model

        self.embedding = nn.Linear(f_in, d_model)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.fc_out(x)
        return x


class MaskedTemporalTransformer(nn.Module):
    """Encoder for masked timestep prediction. Learns dynamics + latent for interpretation.

    Deterministic extraction: pass mask=torch.zeros(B,T,dtype=bool) for no-masking
    to get consistent latents per timestep for downstream HMM analysis.
    """

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        value_mask_ratio: float = 0.0,
        return_layer_mean: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.f_in = f_in
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.value_mask_ratio = value_mask_ratio
        self.return_layer_mean = return_layer_mean
        self.causal = causal
        dim_feedforward = 2 * d_model

        self.embedding = nn.Linear(f_in, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, f_in)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (B, T, F) time series
            mask: (B, T) bool, True = timestep masked. If None, create random timestep mask.
        Returns:
            recon: (B, T, F)
            latent: (B, T, d_model)
            timestep_mask: (B, T) bool
            value_mask: (B, T, F) bool or None when value_mask_ratio=0
        """
        B, T, F = x.shape
        if mask is None:
            timestep_mask = create_timestep_mask(B, T, self.mask_ratio, x.device)
        else:
            timestep_mask = mask

        value_mask: torch.Tensor | None = None
        if self.value_mask_ratio > 0:
            value_mask = create_value_mask(
                B, T, F, self.value_mask_ratio, x.device, exclude_timestep_mask=timestep_mask
            )
            x = x.masked_fill(value_mask, 0.0)

        emb = self.embedding(x)
        emb = emb + (timestep_mask.unsqueeze(-1) * (self.mask_token - emb))

        if self.return_layer_mean:
            layer_outs: list[torch.Tensor] = []
            h = emb
            for layer in self.layers:
                h = layer(h, is_causal=self.causal)
                layer_outs.append(h)
            latent = self.norm(torch.stack(layer_outs, dim=0).mean(dim=0))
        else:
            h = emb
            for layer in self.layers:
                h = layer(h, is_causal=self.causal)
            latent = self.norm(h)
        recon = self.reconstruction_head(latent)
        return recon, latent, timestep_mask, value_mask


class PatchEmbedding(nn.Module):
    """Converts (B, T, F) time series into non-overlapping patch tokens (B, N, d_model).

    Each patch groups P consecutive timesteps across all F features into a single token,
    which is then linearly projected to d_model. T must be divisible by patch_len.
    """

    def __init__(self, f_in: int, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.f_in = f_in
        self.proj = nn.Linear(patch_len * f_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, N, d_model) where N = T // patch_len."""
        B, T, F = x.shape
        P = self.patch_len
        assert T % P == 0, f"Sequence length {T} must be divisible by patch_len {P}"
        N = T // P
        x = x.reshape(B, N, P * F)
        return self.proj(x)


def create_patch_mask(
    batch_size: int, n_patches: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Random mask over patches. Returns bool mask (B, N), True = masked (predict)."""
    n_mask = max(1, int(n_patches * mask_ratio))
    noise = torch.rand(batch_size, n_patches, device=device)
    _, indices = noise.topk(n_mask, dim=1)
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask


class PatchedMaskedTemporalTransformer(nn.Module):
    """Patch-based masked temporal transformer for learning latent representations.

    Input (B, T, F) is divided into non-overlapping patches of length P, producing
    N = T/P patch tokens. Masking operates at patch level — entire patches are replaced
    with a learnable mask token. Reconstruction maps back to full (B, T, F) resolution.

    The latent output is (B, N, d_model) at patch resolution, where each token captures
    local temporal dynamics within P timesteps and attention models inter-patch dependencies.
    """

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        patch_len: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        value_mask_ratio: float = 0.0,
        return_layer_mean: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.f_in = f_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        self.value_mask_ratio = value_mask_ratio
        self.return_layer_mean = return_layer_mean
        self.causal = causal
        dim_feedforward = 2 * d_model

        self.patch_embedding = PatchEmbedding(f_in, patch_len, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, patch_len * f_in)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (B, T, F) time series
            mask: (B, N) bool patch mask, True = masked. If None, create random.
        Returns:
            recon: (B, T, F) full-resolution reconstruction
            latent: (B, N, d_model) patch-level latent
            timestep_mask: (B, T) bool expanded to timestep level for loss
            value_mask: (B, T, F) bool or None
        """
        B, T, F = x.shape
        P = self.patch_len
        assert T % P == 0, f"Sequence length {T} must be divisible by patch_len {P}"
        N = T // P

        if mask is None:
            patch_mask = create_patch_mask(B, N, self.mask_ratio, x.device)
        else:
            patch_mask = mask

        value_mask: torch.Tensor | None = None
        if self.value_mask_ratio > 0:
            timestep_exclude = patch_mask.repeat_interleave(P, dim=1)
            value_mask = create_value_mask(
                B, T, F, self.value_mask_ratio, x.device,
                exclude_timestep_mask=timestep_exclude,
            )
            x = x.masked_fill(value_mask, 0.0)

        emb = self.patch_embedding(x)  # (B, N, d_model)
        emb = emb + (patch_mask.unsqueeze(-1) * (self.mask_token - emb))

        if self.return_layer_mean:
            layer_outs: list[torch.Tensor] = []
            h = emb
            for layer in self.layers:
                h = layer(h, is_causal=self.causal)
                layer_outs.append(h)
            latent = self.norm(torch.stack(layer_outs, dim=0).mean(dim=0))
        else:
            h = emb
            for layer in self.layers:
                h = layer(h, is_causal=self.causal)
            latent = self.norm(h)

        recon_patches = self.reconstruction_head(latent)  # (B, N, P*F)
        recon = recon_patches.reshape(B, N, P, F).reshape(B, T, F)
        timestep_mask = patch_mask.repeat_interleave(P, dim=1)  # (B, T)
        return recon, latent, timestep_mask, value_mask


class PatchedMaskedPredictiveTemporalTransformer(PatchedMaskedTemporalTransformer):
    """Patched transformer with multi-step-ahead prediction at patch level.

    Each patch token predicts the next K patches. Returns pred_next (B, T, K, F)
    by expanding patch-level predictions back to timestep resolution.
    """

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        patch_len: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        value_mask_ratio: float = 0.0,
        n_predict_steps: int = 3,
        return_layer_mean: bool = False,
        causal: bool = False,
    ):
        super().__init__(
            f_in=f_in, d_model=d_model, nhead=nhead, num_layers=num_layers,
            patch_len=patch_len, dropout=dropout, mask_ratio=mask_ratio,
            value_mask_ratio=value_mask_ratio, return_layer_mean=return_layer_mean,
            causal=causal,
        )
        self.n_predict_steps = n_predict_steps
        # Each patch predicts next K patches: K * P * F values
        self.prediction_head = nn.Linear(d_model, n_predict_steps * patch_len * f_in)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        recon, latent, timestep_mask, value_mask = super().forward(x, mask)
        B, T, F = x.shape
        P = self.patch_len
        N = T // P
        K = self.n_predict_steps

        pred_flat = self.prediction_head(latent)  # (B, N, K*P*F)
        # Reshape: each patch n predicts K future patches -> expand to timestep level
        # (B, N, K, P, F) -> repeat each patch P times along T -> (B, T, K, F)
        pred_patches = pred_flat.reshape(B, N, K, P, F)
        # Expand to timestep level: for timestep t in patch n, predict K*P future timesteps
        # grouped as K steps of P timesteps each. To match predictive_loss (B, T, K, F),
        # we interleave: timestep t predicts t+1, t+2, ..., t+K
        # Since all timesteps in a patch share the same latent, we repeat.
        pred_next = pred_patches.repeat_interleave(P, dim=1)  # (B, T, K, P, F)
        # Collapse the inner P dim: for step k, predict the k-th future patch center timestep
        # Take the mean over the P timesteps within each predicted patch
        pred_next = pred_next.mean(dim=3)  # (B, T, K, F)
        return recon, latent, timestep_mask, value_mask, pred_next


def patched_predictive_loss(
    pred_next: torch.Tensor,
    x: torch.Tensor,
    n_predict_steps: int,
    patch_len: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predictive loss for patched models. Operates at patch level for clean alignment.

    pred_next: (B, T, K, F) from PatchedMaskedPredictiveTemporalTransformer.
    Computes loss using patch-averaged targets for each future step k:
    target for step k from patch n = mean of x over timesteps in patch (n+k+1).
    """
    B, T, n_f = x.shape
    P = patch_len
    N = T // P
    K = n_predict_steps
    if N <= K:
        return pred_next.new_zeros(1).squeeze()

    # Average x into patch-level representations: (B, N, n_f)
    x_patches = x.reshape(B, N, P, n_f).mean(dim=2)

    valid_len = N - K
    # Predictions at patch level: take one timestep per patch (e.g., first)
    pred_patch = pred_next[:, ::P, :, :]  # (B, N, K, n_f)
    pred_valid = pred_patch[:, :valid_len, :, :]  # (B, valid_len, K, n_f)

    # Targets: for patch n, step k, target is x_patches[:, n+1+k, :]
    target = torch.stack(
        [x_patches[:, 1 + k : 1 + k + valid_len, :] for k in range(K)], dim=2
    )  # (B, valid_len, K, n_f)

    if valid_mask is None:
        return F.mse_loss(pred_valid, target)

    # valid_mask (B, T) -> patch level (B, N): a patch is valid if all its timesteps are valid
    vm_patches = valid_mask.reshape(B, N, P).all(dim=2)  # (B, N)
    pred_pos_valid = vm_patches[:, :valid_len].unsqueeze(2).unsqueeze(3)
    target_pos_valid = torch.stack(
        [vm_patches[:, 1 + k : 1 + k + valid_len] for k in range(K)], dim=2
    ).unsqueeze(3)
    valid_btk = (pred_pos_valid & target_pos_valid).squeeze(3)
    if not valid_btk.any():
        return pred_next.new_zeros(1).squeeze()
    return F.mse_loss(pred_valid[valid_btk], target[valid_btk])


class MaskedPredictiveTemporalTransformer(MaskedTemporalTransformer):
    """Masked transformer with multi-step-ahead prediction head. Returns (recon, latent, mask, pred_next)."""

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        value_mask_ratio: float = 0.0,
        n_predict_steps: int = 3,
        return_layer_mean: bool = False,
        causal: bool = False,
    ):
        super().__init__(
            f_in=f_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            mask_ratio=mask_ratio,
            value_mask_ratio=value_mask_ratio,
            return_layer_mean=return_layer_mean,
            causal=causal,
        )
        self.n_predict_steps = n_predict_steps
        self.prediction_head = nn.Linear(d_model, n_predict_steps * f_in)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        recon, latent, timestep_mask, value_mask = super().forward(x, mask)
        B, T, F = x.shape
        K = self.n_predict_steps
        pred_flat = self.prediction_head(latent)
        pred_next = pred_flat.view(B, T, K, F)
        return recon, latent, timestep_mask, value_mask, pred_next


def predictive_loss(
    pred_next: torch.Tensor,
    x: torch.Tensor,
    n_predict_steps: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on valid next-step predictions. pred_next (B,T,K,F), x (B,T,F). Only t in 0..T-K used.
    If valid_mask (B,T) is given, only positions where valid_mask is True contribute."""
    B, T, n_f = x.shape
    K = n_predict_steps
    if T <= K:
        return pred_next.new_zeros(1).squeeze()
    valid_len = T - K
    pred_valid = pred_next[:, :valid_len, :, :]
    target = torch.stack([x[:, 1 + k : 1 + k + valid_len, :] for k in range(K)], dim=2)
    if valid_mask is None:
        return F.mse_loss(pred_valid, target)
    # (B, valid_len) for pred pos; for each k, target at t+1+k: (B, valid_len)
    pred_pos_valid = valid_mask[:, :valid_len].unsqueeze(2).unsqueeze(3)
    target_pos_valid = torch.stack(
        [valid_mask[:, 1 + k : 1 + k + valid_len] for k in range(K)], dim=2
    ).unsqueeze(3)
    valid_btk = (pred_pos_valid & target_pos_valid).squeeze(3)
    if not valid_btk.any():
        return pred_next.new_zeros(1).squeeze()
    return F.mse_loss(pred_valid[valid_btk], target[valid_btk])


def masked_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE on masked positions only. pred/target (B,T,F), mask (B,T) bool."""
    pred_m = pred[mask]
    target_m = target[mask]
    return F.mse_loss(pred_m, target_m)


def value_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    value_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on value-masked (b,t,f) positions. value_mask (B,T,F), optional valid_mask (B,T)."""
    if valid_mask is not None:
        value_mask = value_mask & valid_mask.unsqueeze(2)
    if not value_mask.any():
        return pred.new_zeros(1).squeeze()
    return F.mse_loss(pred[value_mask], target[value_mask])


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    unmasked_weight: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on masked; weighted MSE on unmasked for pass-through regularization. unmasked_weight clamped to avoid 0.
    If valid_mask (B,T) is given, only positions where valid_mask is True contribute."""
    unmasked_weight = max(1e-6, unmasked_weight)
    if valid_mask is not None:
        masked_pos = mask & valid_mask
        unmasked_pos = ~mask & valid_mask
    else:
        masked_pos = mask
        unmasked_pos = ~mask
    if masked_pos.any():
        masked_loss = F.mse_loss(pred[masked_pos], target[masked_pos])
    else:
        masked_loss = pred.new_zeros(1).squeeze()
    if unmasked_pos.any():
        unmasked_loss = F.mse_loss(pred[unmasked_pos], target[unmasked_pos])
    else:
        unmasked_loss = pred.new_zeros(1).squeeze()
    return masked_loss + unmasked_weight * unmasked_loss


if __name__ == "__main__":
    # RoPE rotation test
    x = torch.randn(4, 128, 24)
    rope_embd = RotaryPositionalEmbeddings(24)(x)
    print("RoPE rotation:", rope_embd.shape)

    # RoPE attention test
    attn = RoPEMultiHeadAttention(d_model=64, nhead=2)
    h = torch.randn(4, 128, 64)
    out, _ = attn(h)
    print("RoPE attention:", out.shape)

    out_w, weights = attn(h, need_weights=True)
    print("Attention weights:", weights.shape)

    # Masked pretraining example
    model = MaskedTemporalTransformer(f_in=24, d_model=64, nhead=2, num_layers=4, mask_ratio=0.15)
    recon, latent, timestep_mask, value_mask = model(x)
    loss = masked_reconstruction_loss(recon, x, timestep_mask)
    print("Recon:", recon.shape, "Latent:", latent.shape, "Loss:", loss.item())

    # Predictive model test
    pred_model = MaskedPredictiveTemporalTransformer(f_in=24, d_model=64, nhead=2, num_layers=4, causal=True)
    recon, latent, mask, vmask, pred_next = pred_model(x)
    print("Predictive recon:", recon.shape, "pred_next:", pred_next.shape)

    # Patched model tests
    print("\n--- Patched models ---")
    patch_len = 8
    patched = PatchedMaskedTemporalTransformer(f_in=24, d_model=64, nhead=2, num_layers=4, patch_len=patch_len)
    recon_p, latent_p, tmask_p, vmask_p = patched(x)
    n_patches = 128 // patch_len
    print(f"Patched recon: {recon_p.shape}, latent: {latent_p.shape} (N={n_patches} patches)")
    print(f"  timestep_mask: {tmask_p.shape}, masked timesteps: {tmask_p[0].sum().item()}")
    loss_p = masked_reconstruction_loss(recon_p, x, tmask_p)
    print(f"  Patched recon loss: {loss_p.item():.4f}")

    patched_pred = PatchedMaskedPredictiveTemporalTransformer(
        f_in=24, d_model=64, nhead=2, num_layers=4, patch_len=patch_len, causal=True
    )
    recon_pp, latent_pp, tmask_pp, vmask_pp, pred_next_pp = patched_pred(x)
    print(f"Patched predictive recon: {recon_pp.shape}, pred_next: {pred_next_pp.shape}")
    ploss = patched_predictive_loss(pred_next_pp, x, 3, patch_len)
    print(f"  Patched predictive loss: {ploss.item():.4f}")
