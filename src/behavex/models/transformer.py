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


def create_timestep_mask(
    batch_size: int, seq_len: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Random mask over timesteps. Returns bool mask True = masked (predict)."""
    n_mask = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(seq_len, device=device)[:n_mask]
        mask[i, idx] = True
    return mask


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

        self.rms_norm = RMSNorm(d_model)
        self.embedding = nn.Linear(f_in, d_model)
        self.pos_encoder = RotaryPositionalEmbeddings(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='relu',
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.rms_norm(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x


class MaskedTemporalTransformer(nn.Module):
    """Encoder for masked timestep prediction. Learns dynamics + latent for interpretation."""

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.f_in = f_in
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        dim_feedforward = 2 * d_model

        self.embedding = nn.Linear(f_in, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = RotaryPositionalEmbeddings(d_model)
        self.norm = RMSNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Linear(d_model, f_in)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, F) time series
            mask: (B, T) bool, True = masked. If None, create random mask.
        Returns:
            recon: (B, T, F) predicted values
            latent: (B, T, d_model) encoder hidden states for interpretation
            mask: (B, T) bool, used mask (for loss)
        """
        B, T, F = x.shape
        if mask is None:
            mask = create_timestep_mask(B, T, self.mask_ratio, x.device)

        # Embed and substitute mask tokens
        emb = self.embedding(x)
        emb = emb + (mask.unsqueeze(-1) * (self.mask_token - emb))
        emb = self.pos_encoder(emb)
        emb = self.norm(emb)

        latent = self.encoder(emb)
        recon = self.reconstruction_head(latent)
        return recon, latent, mask


def masked_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE on masked positions only. pred/target (B,T,F), mask (B,T) bool."""
    pred_m = pred[mask]
    target_m = target[mask]
    return F.mse_loss(pred_m, target_m)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    unmasked_weight: float = 0.0,
) -> torch.Tensor:
    """MSE on masked; optional weighted MSE on unmasked for pass-through regularization."""
    masked_loss = F.mse_loss(pred[mask], target[mask])
    if unmasked_weight <= 0:
        return masked_loss
    unmasked = ~mask
    unmasked_loss = F.mse_loss(pred[unmasked], target[unmasked])
    return masked_loss + unmasked_weight * unmasked_loss


if __name__ == "__main__":
    # RoPE test
    x = torch.randn(4, 128, 24)
    rope_embd = RotaryPositionalEmbeddings(24)(x)
    print("RoPE:", rope_embd.shape)

    # Masked pretraining example
    model = MaskedTemporalTransformer(f_in=24, d_model=64, nhead=2, num_layers=4, mask_ratio=0.15)
    recon, latent, mask = model(x)
    loss = masked_reconstruction_loss(recon, x, mask)
    print("Recon:", recon.shape, "Latent:", latent.shape, "Loss:", loss.item())
