import torch
import numpy as np
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from basemodels.inr2vec.pcd_retrieval_and_classification.pcd import compute_udf_from_pcd, farthest_point_sampling, random_point_sampling, shuffle_pcd


def get_inr_param_count_from_shapes(weight_shapes, bias_shapes) -> int:
    total = 0
    for shape in weight_shapes:
        total += np.prod(shape)
    for shape in bias_shapes:
        total += np.prod(shape)
    return int(total)


def unflatten_to_wsfeat(flat, weight_shapes, bias_shapes, device='cuda') -> WeightSpaceFeatures:
    """
    Unflatten (batch, total_params) tensor back to WeightSpaceFeatures.

    Args:
        flat: (batch, total_params) tensor
        weight_shapes: List of shape tuples (excluding batch), e.g., [(1, 32, 2), (1, 32, 32), ...]
        bias_shapes: List of shape tuples (excluding batch), e.g., [(1, 32), (1, 32), ...]
    """
    batch_size = flat.shape[0]
    weights, biases, offset = [], [], 0

    for shape in weight_shapes:
        numel = int(np.prod(shape))
        w = flat[:, offset:offset + numel].reshape(batch_size, *shape)
        weights.append(w)
        offset += numel

    for shape in bias_shapes:
        numel = int(np.prod(shape))
        b = flat[:, offset:offset + numel].reshape(batch_size, *shape)
        biases.append(b)
        offset += numel

    return WeightSpaceFeatures(tuple(weights), tuple(biases)).to(device)



class SurrogateAttacker(torch.nn.Module):
    def __init__(self, is_i2i, victim, surrogate, weight_shapes=None, bias_shapes=None, is_3d=False):
        super(SurrogateAttacker, self).__init__()
        self.surrogate = surrogate
        self.victim = victim
        self.weight_shapes = weight_shapes
        self.bias_shapes = bias_shapes
        self.is_3d = is_3d
        self.is_i2i = is_i2i

    def forward(self, src):
        batch_size = src.shape[0]

        # flat_imgs = imgs.permute(0, 2, 3, 1).contiguous()
        # flat_imgs = flat_imgs.view(batch_size, -1)

        if self.is_3d:
            pcd = farthest_point_sampling(src, 2048)
            flat_x = pcd.reshape(batch_size, -1)
        else:
            flat_x = src.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)


        hid = self.surrogate(flat_x)
        
        if self.is_i2i:
            wsfeat = unflatten_to_wsfeat(
                hid,
                self.weight_shapes,
                self.bias_shapes,
            )
            logits = self.victim(wsfeat)
        else:
            if self.victim.model_type == "nft":
                hid = hid.view(batch_size, 16, -1)
            logits = self.victim.clf(hid)
        return logits


