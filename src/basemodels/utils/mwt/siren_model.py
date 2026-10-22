
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange
import math
import random

device = 'cuda'


def layer_(ch_in, ch_out, index, dev, omega=None, fourier=False):
    assert omega is not None, 'omega must be specified'

    w_ = torch.zeros(size=(ch_in, ch_out,), device=dev) # [CI, CO]
    b_ = torch.zeros(size=(ch_out,), device=dev) # [CO]

    if (index == 0):
        v = 1. / ch_in
    else:
        v = np.sqrt(6 / ch_in)
    torch.nn.init.uniform_(w_, -v, v)

    train = True # (index > 0)
    w_ = nn.Parameter(w_, requires_grad=train)
    b_ = nn.Parameter(b_, requires_grad=train)
    return [ w_, b_ ] # [CI, CO], [G, CI//G, CO//G] or [CO]


def get_pass_indices(siren):
    r = []
    for i, p in enumerate(siren.sirens):
        r += [1] * p.numel()
    indices = torch.tensor(r, device=p.device).nonzero().squeeze()
    return indices


class SirenModel(nn.Module):

    def __init__(self, ch_in, ch_hiddens, ch_out, omega=30.0):
        super(SirenModel, self).__init__()
        self.ch_out = ch_out
        self.omega = omega

        # initialize an initial SIREN weight list
        siren_base = []

        dim = ch_hiddens[0]
        depth = len(ch_hiddens)
        self.depth = depth

        siren_base += layer_(ch_in, dim, index=0, dev=device, omega=omega)
        for _ in range(depth):
            siren_base += layer_(dim, dim, index=1, dev=device, omega=omega)

        self.sirens = nn.ParameterList(siren_base)
        self.out_scale = nn.Parameter(0.1 * torch.ones((ch_out,)))
        r_init = lambda size, a, b: a + (b - a) * torch.rand(size=size, device=device)
        ch_hidden = ch_hiddens[0]
        self.out = nn.Linear(dim, ch_out)

        self.pass_indices = get_pass_indices(self)
        # print number of parameters for each named param
        for name, param in self.named_parameters():
            print(name, param.numel())

    def num_params(self):
        return sum([ p.numel() for p in self.sirens ]) # exclude the last bias and weight # [:-2]


    def forward(self, x, omega=0, weight_offset=None):

        # weight_offset as [N, E]
        assert len(x.shape) == 3, 'x should be [N, H*W, CI]'
        assert len(weight_offset.shape) == 2, 'weight_offset should be [N, E]'
        assert weight_offset.shape[1] == self.num_params(), 'weight_offset should have same number of elements as parameters'
        assert weight_offset.shape[0] == x.shape[0], 'weight_offset should have same batch size as x'
        N, HW = x.shape[0:2]

        WC = self.num_params()
        N = x.shape[0]
        # x is [N, H*W, CI]
        # each weight gets an offset
        shapes = [ p.shape for p in self.sirens ] # [:-2]
        offsets = []
        global_i = 0
        for shape in shapes:
            w = weight_offset[:, global_i:(global_i+shape.numel())]
            w = w.view(*([N] + list(shape)))
            offsets.append(w)
            global_i += shape.numel()
        assert global_i == WC, 'global_i should be equal to WC but %d != %d' % (global_i, WC)         
        assert len(offsets) == len(self.sirens), 'offsets should have same length as self.sirens'

        m = self.sirens # the siren weights and biases
        pairs = [ (m[i], m[i+1]) for i in range(0, len(m)-1, 2) ]

        skip = None
        for i, (w, b) in enumerate(pairs):
            first = (i == 0)
            last = (i == (len(pairs) - 1))

            off_w = offsets[i*2] # [N, CI, CO]
            off_b = offsets[i*2 + 1] # [N, CO]
            off_b = off_b.unsqueeze(1) # [N, 1, CO]
            w = w.view(1, *w.shape) # [1, CI, CO]
            w = w + off_w # [N, CI, CO]

            x = torch.matmul(x, w) # [N, H*W, CO]
            b = b.view(1, 1, -1) # [1, 1, CO] share over all H*W pixels
            b = b + off_b
            x = x + b # [N, H*W, CO]

            if first:
                x = torch.sin(self.omega * x)
            else:
                x = torch.sin(x)

        x = self.out(x)
        x = x * self.out_scale # [N, H*W, CO]

        assert len(x.shape) == 3, 'x should be [N, H*W, CO]'
        assert x.shape[0] == N, 'batch size should be same as input'
        assert x.shape[1] == HW, 'H*W should be same as input'
        assert x.shape[2] == self.ch_out, 'CO should be same as output scale'

        return x
