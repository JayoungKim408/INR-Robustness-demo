
import torch
import torch.nn as nn


class MetaSGD(nn.Module):

    def __init__(self, conf, PI):
        super(MetaSGD, self).__init__()
        self.conf = conf

        r_init = lambda size, a, b: a + (b - a) * torch.rand(size=size)
        self.meta_lrs_w = nn.ParameterList()
        for _ in range(1 if conf['shared_lr'] else conf['inner_steps']):
            self.meta_lrs_w += [ nn.Parameter(r_init([1, PI], 0.1, 1.0)) ]

    def forward(self, j):
        if self.conf['shared_lr']:
            use_lr = self.meta_lrs_w[0] # [1, WC]
        else:
            use_lr = self.meta_lrs_w[j] # [1, WC]
        return use_lr
