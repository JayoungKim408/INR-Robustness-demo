
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class WT(nn.Module):

    def __init__(self, dim, conf):
        super(WT, self).__init__()
        self.conf = conf
            
        self.project_in = nn.Linear(dim+1, dim)
        self.classifier = nn.Sequential(
            *[ nn.Sequential(SelfAttention(dim=dim, layer_scale=0.1), Block(dim=dim, mult=conf['mlp_mult'], layer_scale=0.1)) \
                    for _ in range(conf['classifier_depth']) ]
        )
        self.to_logits = nn.Linear(dim, conf['class_count'])
        depth = len(conf['siren_dim'])
        self.base = nn.Parameter(torch.zeros(size=(1, depth * dim, dim+1)))

    def forward(self, l):
        # l is a list of tuples of weights and biases for each layer
        all = []
        for w, b in l:
            assert len(w.shape) == 3, 'w should be 3D'
            assert len(b.shape) == 2, 'b should be 2D'
            # w as [N, CI, CO] and b as [N, CO]
            b = b.unsqueeze(1) # [N, 1, CO]
            w = torch.cat([w, b], dim=1) # [N, CI+1, CO]
            w = rearrange(w, 'n ci co -> n co ci') # [N, CO, CI+1]
            all += [ w ]
        all = torch.cat(all, dim=1) # [N, CO, CI+1]
        all = (all + self.base) # [N, CO, CI+1]
        if self.conf['modulation_scale'] > 0:
            all = (all * self.conf['modulation_scale']) # [N, CO, CI+1]
        else:
            all = F.layer_norm(all, all.shape[1:]) # [N, CO, CI+1]
        all = self.project_in(all) # [N, CO, CI]
        all = self.classifier(all) # [N, 1+CO, CI]
        all = all.mean(dim=1) # [N, CI]
        return self.to_logits(all) # [N, 10]



class Block(nn.Module):

    def __init__(self, dim, mult, layer_scale):
        super(Block, self).__init__()

        self.m = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
        self.scale = nn.Parameter(layer_scale * torch.ones((dim,)))

    def forward(self, x):
        return (x + self.scale * self.m(x))


class SelfAttention(nn.Module):

    def __init__(self, dim, layer_scale):
        super().__init__()
        self.dim = dim

        self.head_dim = 64
        self.heads = dim // self.head_dim
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(self.dim)
        self.to_qkv = nn.Linear(self.dim, 3*self.heads*self.head_dim, bias=False)
        self.to_out = nn.Linear(self.heads*self.head_dim, self.dim)

        self.layer_scale = nn.Parameter(layer_scale * torch.ones((self.dim,)))

    def forward(self, x):
        # print('input act peek', x[0, 0, :5], 'num tokens', x.shape[1])
        skip = x

        # x as [N, K, C]
        N, K, C = x.shape
        x = self.norm(x)
        # print('post norm peek', x[0, 0, :5])
        qkv = self.to_qkv(x) # [N, K, 3*H]
        qkv = rearrange(qkv, 'n k (qkv h d) -> qkv n h k d', h=self.heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2] # [N, H, K, D]

        att = torch.einsum('n h i d, n h j d -> n h i j', q, k) # [N, H, K, K]
        att = att * self.scale
        att = F.softmax(att, dim=-1) # [N, H, K, K]

        out = torch.einsum('n h i j, n h j d -> n h i d', att, v) # [N, H, K, D]
        out = rearrange(out, 'n h k d -> n k (h d)') # [N, K, H*D]
        out = self.to_out(out) # [N, K, C]
        out = (out * self.layer_scale)

        # print('out range', out.min().item(), out.max().item(), 'peek', out[0, 0, :5])

        return out + skip
