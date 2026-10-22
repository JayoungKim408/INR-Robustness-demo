
import copy
import torch
import wandb
import subprocess

import numpy as np
from einops import rearrange
from PIL import Image as PILImage


def compute_psnr(target_rgb, pred_rgb):
    N = target_rgb.shape[0]
    assert len(target_rgb.shape) == 3, 'target_rgb should be [N, H*W, 3]'
    assert target_rgb.shape == pred_rgb.shape, 'target_rgb and pred_rgb should have same shape'
    # first make sure both are scaled [0, 1]
    target_rgb = (target_rgb + 1.0) / 2.0 # [N, H*W, 3]
    assert target_rgb.min() >= 0 and target_rgb.max() <= 1, 'target_rgb should be in [0, 1]'
    pred_rgb = (pred_rgb + 1.0) / 2.0 # [N, H*W, 3]
    pred_rgb = pred_rgb.clamp(min=0, max=1)
    assert len(pred_rgb.shape) == 3, 'pred_rgb should be [N, H*W, 3]'
    assert pred_rgb.shape == target_rgb.shape, 'pred_rgb and target_rgb should have same shape'
    mse = ((target_rgb - pred_rgb) ** 2).mean(dim=[1, 2]) # [N]
    assert mse.shape == (N,), 'mse should be [N]'
    assert mse.min() >= 0, 'mse should be non-negative'
    psnr = 10 * torch.log10(1.0 / mse) # [N]
    assert psnr.shape == (N,), 'psnr should be [N]'
    return psnr # [N]

def get_mgrid(height, width, dev):
    xs = torch.linspace(-1, +1, steps=width, device=dev)
    ys = torch.linspace(-1, +1, steps=height, device=dev)
    return torch.stack(torch.meshgrid([ys, xs], indexing='ij'), dim=-1).view(-1, 2) # [H*W, XY=2]

def to_wandb_image(pixels, caption='No caption provided'):
    if pixels.shape[2] == 1:
        pixels = pixels.repeat(1, 1, 3)
    pixels = pixels.clamp(min=-1, max=+1)
    pixels = (pixels + 1) / 2 * 255
    pixels = pixels.cpu().detach() # .permute(1, 2, 0)
    pixels = pixels.numpy().astype(np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    return wandb.Image(pil_image, caption)

def sample(samples, coords_flat, img_flat):
    N, HW, RGB = img_flat.shape
    pred_w = torch.ones((N, HW), device=img_flat.device)
    indices = torch.multinomial(pred_w, samples) # [1, INDEX]
    indices_xy = indices.unsqueeze(-1).expand(-1, -1, 2) # [N, 10000, 2] - repeat over XY
    indices_rgb = indices.unsqueeze(-1).expand(-1, -1, RGB) # [N, 10000, 3] - repeat over RGB
    use_coords = torch.gather(coords_flat, 1, indices_xy) # [N, 10000, 2]
    use_image = torch.gather(img_flat, 1, indices_rgb) # [N, 10000, 3]
    return use_coords, use_image, indices


def preview_img(siren, coords_flat, weight_offset, indices, preview_dict, device, H, W):
    if preview_dict is not None:
        siren = copy.deepcopy(siren)
        weight_offset = weight_offset.clone()
        with torch.no_grad():
            pred_rgb = siren(coords_flat[0:1], weight_offset=weight_offset[0:1]) # [1, H*W, 3]
            if 'images' not in preview_dict:
                preview_dict['images'] = []
            preview_dict['images'].append(rearrange(pred_rgb[0], '(h w) c -> h w c', h=H, w=W))

            grad0 = -1 * torch.ones((H*W, 3), device=device)
            grad0[indices[0], :] = -1.0
            grad0[indices[0], 0] = 1.0
            if 'grads' not in preview_dict:
                preview_dict['grads'] = []
            preview_dict['grads'].append(rearrange(grad0, '(h w) c -> h w c', h=H, w=W))


def get_mem():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error running nvidia-smi: {result.stderr.strip()}")
    memory_values = result.stdout.strip().split("\n")
    if not memory_values:
        raise ValueError("No GPU memory usage data found.")
    return int(memory_values[0]) # first GPU memory usage