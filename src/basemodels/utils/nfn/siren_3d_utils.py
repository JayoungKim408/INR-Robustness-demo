"""
batch_siren_3d.py

Batch rendering function for 3D ShapeNet INRs (UDF).
Evaluates INR on a fixed 3D grid in [-1, 1]^3.

Usage:
    from batch_siren_3d import get_batch_siren_3d
    batch_render, init_params = get_batch_siren_3d(siren_config=None, resolution=32, device="cuda")
    
    # params: tuple of tensors, each (batch_size, ...)
    udf_values = batch_render(params)  # (batch_size, resolution^3)
"""

import math
import numpy as np
import torch
from torch import nn
from torch.func import vmap, functional_call

# Import your Siren class - adjust path as needed
try:
    from basemodels.utils.nfn.siren_utils import Siren, get_mgrid
except ImportError:
    from siren_utils import Siren, get_mgrid


SHAPENET_SIREN_KWARGS = {
    "first_omega_0": 30,
    "hidden_features": 128,
    "hidden_layers": 1,  # siren_utils convention: 1 hidden layer = 3 linear layers total
    "hidden_omega_0": 30.0,
    "in_features": 3,
    "out_features": 1,
    "outermost_linear": True,
}


def get_mgrid_3d(resolution, device="cuda"):
    """Generate a fixed 3D grid in [-1, 1]^3."""
    coords = get_mgrid(resolution, dim=3).to(device)  # (resolution^3, 3)
    return coords


def get_batch_siren_3d(siren_config=None, resolution=32, device="cuda"):
    """
    Get batch rendering function for 3D ShapeNet INRs.
    
    Args:
        siren_config: Optional dict overriding SIREN architecture
        resolution: Grid resolution per dimension (total points = resolution^3)
        device: Device
        
    Returns:
        (batch_render_fn, initial_params)
        - batch_render_fn: takes params tuple -> (batch_size, resolution^3, 1)
        - initial_params: tuple of initial parameter tensors
    """
    if siren_config:
        siren_kwargs = {
            "first_omega_0": siren_config.get("first_omega_0", 30),
            "hidden_features": siren_config.get("hidden_features", 128),
            "hidden_layers": siren_config.get("hidden_layers", 1),
            "hidden_omega_0": siren_config.get("hidden_omega_0", 30.0),
            "in_features": 3,
            "out_features": 1,
            "outermost_linear": True,
        }
    else:
        siren_kwargs = SHAPENET_SIREN_KWARGS.copy()
    
    siren = Siren(**siren_kwargs).to(device)
    
    param_names = list(dict(siren.named_parameters()).keys())
    initial_params = tuple(p.clone() for p in siren.parameters())
    
    coords = get_mgrid_3d(resolution, device)  # (resolution^3, 3)
    
    def func_inp(params_tuple):
        params_dict = dict(zip(param_names, params_tuple))
        values = functional_call(siren, params_dict, (coords,))[0]  # (resolution^3, 1)
        return values.squeeze(-1)  # (resolution^3,)
    
    return vmap(func_inp, in_dims=(0,)), initial_params


def get_spatial_batch_siren_3d(siren_config=None, resolution=32, device="cuda"):
    """
    Spatial version for 3D ShapeNet INRs.
    Each spatial location gets its own set of INR parameters.
    
    Returns:
        (batch_render_fn, initial_params)
    """
    if siren_config:
        siren_kwargs = {
            "first_omega_0": siren_config.get("first_omega_0", 30),
            "hidden_features": siren_config.get("hidden_features", 128),
            "hidden_layers": siren_config.get("hidden_layers", 1),
            "hidden_omega_0": siren_config.get("hidden_omega_0", 30.0),
            "in_features": 3,
            "out_features": 1,
            "outermost_linear": True,
        }
    else:
        siren_kwargs = SHAPENET_SIREN_KWARGS.copy()
    
    siren = Siren(**siren_kwargs).to(device)
    
    param_names = list(dict(siren.named_parameters()).keys())
    initial_params = tuple(p.clone() for p in siren.parameters())
    
    coords = get_mgrid_3d(resolution, device)  # (resolution^3, 3)
    
    def siren_at_coord(params_tuple, coord):
        params_dict = dict(zip(param_names, params_tuple))
        return functional_call(siren, params_dict, (coord[None],))[0].squeeze(0)
    
    batch_siren_at_coord = vmap(siren_at_coord, (0, 0))
    
    def spatial_siren(params):
        # For spatial: params have shape (num_patches, ...) per parameter
        width = int(round(params[0].shape[0] ** (1/3)))  # cube root for 3D
        def reshape(arr):
            return arr.reshape(width, width, width, *arr.shape[1:])
        params = tuple(reshape(p) for p in params)
        
        # Map coords to parameter indices
        param_idcs = torch.round((params[0].shape[0] - 1) * (coords + 1) / 2).long()
        idxd_params = tuple(
            p[param_idcs[:, 0], param_idcs[:, 1], param_idcs[:, 2]] 
            for p in params
        )
        values = batch_siren_at_coord(idxd_params, coords)
        return values.squeeze(-1)  # (resolution^3,)
    
    return vmap(spatial_siren, in_dims=(0,)), initial_params


# Integration with existing siren_utils.py:
# Add these to SIREN_kwargs and the get_batch_siren/get_spatial_batch_siren functions

PATCH_CODE = """
# Add to siren_utils.py:

# 1. Update IMG_shapes:
IMG_shapes = {"mnist": (28, 28, 1), "cifar": (32, 32, 3), "fashion": (28, 28, 1), "shapenet": ()}

# 2. In get_batch_siren, add shapenet handling:
def get_batch_siren(dset_type, siren_config=None, device="cuda"):
    if dset_type == "shapenet":
        from batch_siren_3d import get_batch_siren_3d
        return get_batch_siren_3d(siren_config=siren_config, device=device)
    # ... rest of existing code ...

# 3. In get_spatial_batch_siren, add shapenet handling:
def get_spatial_batch_siren(dset_type, siren_config=None, device="cuda"):
    if dset_type == "shapenet":
        from batch_siren_3d import get_spatial_batch_siren_3d
        return get_spatial_batch_siren_3d(siren_config=siren_config, device=device)
    # ... rest of existing code ...
"""

if __name__ == "__main__":
    # Quick test
    print("Testing 3D batch SIREN...")
    batch_fn, init_params = get_batch_siren_3d(resolution=16, device="cuda")
    
    # Simulate a batch of 4 INRs
    batch_params = tuple(p.unsqueeze(0).expand(4, *p.shape) for p in init_params)
    
    output = batch_fn(batch_params)
    print(f"Output shape: {output.shape}")  # Should be (4, 16^3) = (4, 4096)
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("Test passed!")