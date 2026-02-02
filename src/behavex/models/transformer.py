import torch 
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d:int, base:int=10_000):

        super().__init__()
        self.base = base 
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def build_cache(self, x:torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return 

        seq_len = x.shape[0]
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device) #theta: (10,000^(-2*i/d))

        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) # positition index 
        idx_theta = torch.einsum('n, d-> nd', seq_idx, theta) # calculuates m * theta 
        idx_theta_2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta_2.cos()[:, None, None, :] # Cache for (cos_THETA1, cos_THETA2, cos_THETA3)
        self.sin_cached = idx_theta_2.sin()[:, None, None, :] # Cache for (cos_THETA1, cos_THETA2, cos_THETA3)

    def _neg_half(self, x:torch.Tensor):
        d_2 = self.d //2 
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]
    
    def forward(self, x:torch.Tensor):
        self.build_cache(x)
        neg_half = self._neg_half(x)
        x_rope = (x*self.cos_cached[:x.shape[0]]) + (neg_half * self.sin_cached[:x.shape[0]])
        return x_rope


    


