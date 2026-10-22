import collections
from typing import Any, Dict, List, Union, Iterable

import torch
import torch.nn as nn
from tqdm import tqdm 
import torch.nn.functional as F
from torch import Tensor
# Optional pycarus import - only needed for 3D shape tasks
try:
    from pycarus.learning.models.siren import SIREN
except ImportError:
    SIREN = nn.Module  # Fallback type hint


def get_mlps_batched_params(mlps: List[Union[SIREN, nn.Module]]) -> List[Tensor]:
    params = []
    for i in range(len(mlps)):
        params.append(list(mlps[i].parameters()))

    batched_params = []
    for i in range(len(params[0])):
        p = torch.stack([p[i] for p in params], dim=0)
        p = torch.clone(p.detach())
        p.requires_grad = True
        batched_params.append(p)

    return batched_params


def flatten_mlp_params(sd: Dict[str, Any]) -> Tensor:
    all_params = []
    for k in sd:
        all_params.append(sd[k].view(-1))
    all_params = torch.cat(all_params, dim=-1)
    return all_params


def unflatten_mlp_params(
    params: Tensor,
    sample_sd: Dict[str, Any],
) -> Dict[str, Any]:
    sd = collections.OrderedDict()

    start = 0
    for k in sample_sd:
        end = start + sample_sd[k].numel()
        layer_params = params[start:end].view(sample_sd[k].shape)
        sd[k] = layer_params
        start = end

    return sd


def get_mlp_params_as_matrix(flattened_params: Tensor, sd: Dict[str, Any]) -> Tensor:
    params_shapes = [p.shape for p in sd.values()]
    feat_dim = params_shapes[0][0]
    start = params_shapes[0].numel() + params_shapes[1].numel()
    end = params_shapes[-1].numel() + params_shapes[-2].numel()
    params = flattened_params[start:-end]
    return params.reshape((-1, feat_dim))


def mlp_batched_forward(batched_params: List[Tensor], coords: Tensor, inr_type: str = "siren") -> Tensor:
    """Batched forward pass for INRs.

    Args:
        batched_params: List of batched parameter tensors [w0, b0, w1, b1, ...]
        coords: Input coordinates (batch, num_coords, in_dim)
        inr_type: 'siren' or 'ffn'

    Returns:
        Output tensor (batch, num_coords) or (batch, num_coords, out_dim)
    """
    if inr_type == "ffn":
        return ffn_batched_forward(batched_params, coords)
    else:
        return siren_batched_forward(batched_params, coords)


def siren_batched_forward(batched_params: List[Tensor], coords: Tensor) -> Tensor:
    """Batched forward pass for SIREN INRs."""
    num_layers = len(batched_params) // 2

    f = coords

    for i in range(num_layers):
        weights = batched_params[i * 2]
        biases = batched_params[i * 2 + 1]

        f = torch.bmm(f, weights.permute(0, 2, 1)) + biases.unsqueeze(1)

        if i < num_layers - 1:
            f = torch.sin(30 * f)

    return f.squeeze(-1)


def ffn_batched_forward(batched_params: List[Tensor], coords: Tensor, encoder_B: Tensor) -> Tensor:
    """Batched forward pass for FFN INRs.

    Args:
        batched_params: List of batched MLP parameter tensors [w0, b0, w1, b1, ...]
                        (excludes encoder.B which is passed separately)
        coords: Input coordinates (batch, num_coords, in_dim)
        encoder_B: Fourier feature matrix (batch, num_features, in_dim)

    Returns:
        Output tensor (batch, num_coords) or (batch, num_coords, out_dim)
    """
    if encoder_B is None:
        raise ValueError("encoder_B is required for FFN batched forward")

    # Apply Fourier encoding: γ(x) = [cos(2πBx), sin(2πBx)]
    # coords: (batch, num_coords, in_dim)
    # encoder_B: (batch, num_features, in_dim)
    # proj = coords @ B.T = (batch, num_coords, num_features)
    proj = torch.bmm(coords, encoder_B.permute(0, 2, 1))  # (batch, num_coords, num_features)

    import numpy as np
    f = torch.cat([torch.cos(2 * np.pi * proj),
                   torch.sin(2 * np.pi * proj)], dim=-1)  # (batch, num_coords, 2*num_features)

    # MLP with ReLU activations
    num_layers = len(batched_params) // 2

    for i in range(num_layers):
        weights = batched_params[i * 2]
        biases = batched_params[i * 2 + 1]

        f = torch.bmm(f, weights.permute(0, 2, 1)) + biases.unsqueeze(1)

        if i < num_layers - 1:
            f = torch.relu(f)

    return f.squeeze(-1)


def focal_loss(pred: Tensor, gt: Tensor, alpha: float = 0.1, gamma: float = 3) -> Tensor:
    alpha_w = torch.tensor([alpha, 1 - alpha]).cuda()

    bce_loss = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction="none")
    bce_loss = bce_loss.view(-1)

    gt = gt.type(torch.long)
    at = alpha_w.gather(0, gt.view(-1))
    pt = torch.exp(-bce_loss)
    f_loss = at * ((1 - pt) ** gamma) * bce_loss

    return f_loss.mean()


def get_layer_layout(
    in_features: int = 2,
    out_features: int = 1,
    hidden_features: int = 32,
    hidden_layers: int = 1,
) -> List[int]:
    """
    Get the layer layout for INR2Vec encoder.

    Args:
        in_features: Input dimension (e.g., 2 for coordinates)
        out_features: Output dimension (e.g., 1 for grayscale, 3 for RGB)
        hidden_features: Number of hidden units per layer
        hidden_layers: Number of hidden layers

    Returns:
        List of layer sizes: [in_features, hidden_features, ..., out_features]
    """
    layout = [in_features]
    for _ in range(hidden_layers):
        layout.append(hidden_features)
    layout.append(out_features)
    return layout


def wsfeat_to_inr2vec_matrix(wsfeat, layer_layout: List[int]) -> Tensor:
    """
    Convert WeightSpaceFeatures to INR2Vec matrix format.

    INR2Vec encoder expects a matrix of shape (batch, num_params, feat_dim).
    This function extracts the relevant parameters from WeightSpaceFeatures
    and formats them appropriately.

    Args:
        wsfeat: WeightSpaceFeatures object with weights and biases
        layer_layout: List of layer sizes from get_layer_layout()

    Returns:
        Matrix tensor of shape (batch, -1, feat_dim)
    """
    batch_size = wsfeat.weights[0].shape[0]
    feat_dim = layer_layout[1]  # hidden_features

    # Flatten all params
    all_params = []
    for i, (w, b) in enumerate(zip(wsfeat.weights, wsfeat.biases)):
        # w shape: (batch, out_feat, in_feat)
        # b shape: (batch, out_feat)
        all_params.append(w.reshape(batch_size, -1))
        all_params.append(b.reshape(batch_size, -1))

    flat_params = torch.cat(all_params, dim=1)  # (batch, total_params)

    # For INR2Vec, we need to extract middle layer params and reshape
    # Skip first layer (input) and last layer (output) params
    first_layer_params = layer_layout[0] * layer_layout[1] + layer_layout[1]
    last_layer_params = layer_layout[-2] * layer_layout[-1] + layer_layout[-1]

    middle_params = flat_params[:, first_layer_params:-last_layer_params]

    # Reshape to matrix format
    matrix = middle_params.reshape(batch_size, -1, feat_dim)

    return matrix


def get_class_to_parts(dset_name: str) -> Dict[str, List[int]]:
    shapenet_partseg = {
        "02691156": [0, 1, 2, 3],
        "02773838": [4, 5],
        "02954340": [6, 7],
        "02958343": [8, 9, 10, 11],
        "03001627": [12, 13, 14, 15],
        "03261776": [16, 17, 18],
        "03467517": [19, 20, 21],
        "03624134": [22, 23],
        "03636649": [24, 25, 26, 27],
        "03642806": [28, 29],
        "03790512": [30, 31, 32, 33, 34, 35],
        "03797390": [36, 37],
        "03948459": [38, 39, 40],
        "04099429": [41, 42, 43],
        "04225987": [44, 45, 46],
        "04379243": [47, 48, 49],
    }

    gallery = {"shapenet-partseg": shapenet_partseg}

    return gallery[dset_name]




def progress_bar(iterable: Iterable, desc: str = "", num_cols: int = 60) -> Iterable:
    """Decorate an iterable object using a progress bar.

    Args:
        iterable: the iterable to decorate.
        desc: the description to print. Defaults to "".
        num_cols: The width of the entire output message. Defaults to 60.
        rank: The rank of the current process. Defaults to 0.

    Returns:
        The decorated iterable.
    """
    bar_format = "{percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    if len(desc) > 0:
        bar_format = "{desc}: " + bar_format
    return tqdm(iterable, desc=desc, bar_format=bar_format, ncols=num_cols, leave=False)
