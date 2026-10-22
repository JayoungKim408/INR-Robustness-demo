import math
from typing import Optional

import torch
from rff.layers import GaussianEncoding, PositionalEncoding
from torch import nn


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.init_weights(c, w0)
        self.activation = Sine(w0) if activation is None else activation

    def init_weights(self, c: float, w0: float):
        with torch.no_grad():
            dim = self.dim_in

            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
            self.linear.weight.uniform_(-w_std, w_std)

            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out


class INR(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        n_layers: int = 3,
        up_scale: int = 4,
        out_channels: int = 1,
        pe_features: Optional[int] = None,
        fix_pe=True,
    ):
        super().__init__()
        hidden_dim = in_dim * up_scale

        if pe_features is not None:
            if fix_pe:
                self.layers = [PositionalEncoding(sigma=10, m=pe_features)]
                encoded_dim = in_dim * pe_features * 2
            else:
                self.layers = [
                    GaussianEncoding(
                        sigma=10, input_size=in_dim, encoded_size=pe_features
                    )
                ]
                encoded_dim = pe_features * 2
            self.layers.append(Siren(dim_in=encoded_dim, dim_out=hidden_dim))
        else:
            self.layers = [Siren(dim_in=in_dim, dim_out=hidden_dim)]
        for i in range(n_layers - 2):
            self.layers.append(Siren(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_channels))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + 0.5
