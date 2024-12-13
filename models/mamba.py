import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from typing import Union
import math


class ResidualBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 ):
        super().__init__()

        self.norm = RMSNorm(d_model)

        self.mixer = MambaBlock(d_model)

    def forward(self, x):
        x1 = self.norm(x)  # [n, d]
        x2 = self.mixer(x1)

        output = x2 + x1
        return output


class Mamba(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 n_layer: int = 4,
                 dim_in: int = 2,
                 ):
        super().__init__()

        self.encode = nn.Linear(dim_in, d_model)
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layer):
            self.encoder_layers.append(ResidualBlock(d_model))

        # readout
        self.encoder_norm = RMSNorm(d_model)
        self.decode = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.encode(x)  # [n, d]
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        x = self.decode(x)

        return x


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 bias: bool = False,
                 conv_bias: bool = True,
                 d_conv: int = 4,
                 dt_rank: Union[int, str] = 'auto',
                 d_state: int = 2,
                 ):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias)
        self.d_model = d_model

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
        )

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        x = x.transpose(0, 1)
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_model, self.d_model], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        # output = output.squeeze(0)
        x = x.transpose(0, 1)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        # deltaA = torch.exp(einsum(A, 'd_in n -> b l d_in n'))
        # deltaB = einsum(B, u, 'b l n, b l d_in -> b l d_in n')
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
