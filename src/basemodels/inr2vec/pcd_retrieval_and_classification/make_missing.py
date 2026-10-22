"""
fill_missing_inrs.py

Find missing h5 files from INR fitting and create only those.
Also handles val/test splits if they weren't created.

Usage:
    python fill_missing_inrs.py cfg/inrs_dataset_with_init.yaml
"""

import os
import sys
sys.path.insert(0, "/workspace/INR-Robustness/src")

import yaml
import argparse
from pathlib import Path
from typing import List, Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

from pcd import compute_udf_from_pcd, farthest_point_sampling, random_point_sampling, shuffle_pcd
from transform_pcd import NormalizePcdIntoUnitSphere
from basemodels.utils.nfn.siren_utils import Siren
from basemodels.inr2vec.utils import get_mlps_batched_params, mlp_batched_forward


class H5PcdDataset(Dataset):
    def __init__(self, h5_root, split, transforms=None):
        self.h5_root = Path(h5_root) / split
        self.h5_paths = sorted(self.h5_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.transforms = transforms or []

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        with h5py.File(self.h5_paths[idx], "r") as f:
            pcd = torch.from_numpy(np.array(f["pcd"])).float()
            class_id = torch.tensor(int(f["class_id"][()]), dtype=torch.long)
        original_idx = int(self.h5_paths[idx].stem)
        for t in self.transforms:
            pcd = t(pcd)
        return pcd, class_id, original_idx


def build_mlp(cfg):
    mlp = Siren(
        in_features=3,
        out_features=1,
        hidden_features=cfg["mlp"]["hidden_dim"],
        hidden_layers=cfg["mlp"]["num_hidden_layers"],
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30,
        is_first=cfg.get("is_first", True),
    )
    if cfg.get("bias_zero_init", False):
        for name, param in mlp.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
    return mlp


def find_missing(out_root, split, expected_count):
    split_dir = Path(out_root) / split
    if not split_dir.exists():
        return list(range(expected_count))
    existing = set(int(f.stem) for f in split_dir.glob("*.h5"))
    expected = set(range(expected_count))
    missing = sorted(expected - existing)
    return missing


def fit_and_save(cfg, split, indices_to_create):
    """Fit INRs for specific indices and save."""
    if not indices_to_create:
        print(f"  No missing files for {split}.")
        return

    out_root = Path(cfg["out_root"])
    pcd_root = Path(cfg["pcd_root"])
    transforms = [NormalizePcdIntoUnitSphere()]
    
    dset = H5PcdDataset(pcd_root, split, transforms)
    
    # Map original_idx -> dataset position
    idx_to_pos = {}
    for pos in range(len(dset)):
        original_idx = int(dset.h5_paths[pos].stem)
        idx_to_pos[original_idx] = pos
    
    # Filter to only missing indices that exist in source
    valid_missing = [(idx, idx_to_pos[idx]) for idx in indices_to_create if idx in idx_to_pos]
    
    if not valid_missing:
        print(f"  No valid missing indices for {split}.")
        return
    
    print(f"  Fitting {len(valid_missing)} missing INRs for {split}...")
    
    num_parallel = cfg["num_parallel_mlps"]
    num_steps = cfg["num_steps"]
    lr = float(cfg["lr"])
    num_points_fitting = cfg["num_points_fitting"]
    num_points_pcd = cfg["num_points_pcd"]
    num_queries_on_surface = cfg["num_queries_on_surface"]
    stds = cfg["stds"]
    num_points_per_std = cfg["num_points_per_std"]
    
    # Process in batches
    for batch_start in range(0, len(valid_missing), num_parallel):
        batch_items = valid_missing[batch_start:batch_start + num_parallel]
        bs = len(batch_items)
        
        # Load pcds
        pcds = []
        class_ids = []
        file_indices = []
        for file_idx, pos in batch_items:
            pcd, class_id, orig_idx = dset[pos]
            pcds.append(pcd)
            class_ids.append(class_id)
            file_indices.append(file_idx)
        
        pcds = torch.stack(pcds, dim=0).cuda()
        
        if pcds.shape[1] != num_points_pcd:
            pcds = farthest_point_sampling(pcds, num_points_pcd)
        
        # Generate query points
        coords = []
        labels = []
        for j in range(bs):
            pcd_coords, pcd_labels = compute_udf_from_pcd(
                pcds[j],
                num_queries_on_surface,
                stds,
                num_points_per_std,
                coords_range=(-1, 1),
                convert_to_bce_labels=True,
            )
            coords.append(pcd_coords)
            labels.append(pcd_labels)
        
        coords = torch.stack(coords, dim=0)
        labels = torch.stack(labels, dim=0)
        coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1).cuda()
        coords_and_labels = shuffle_pcd(coords_and_labels)
        
        # Build and train MLPs
        mlps = [build_mlp(cfg).cuda() for _ in range(bs)]
        batched_params = get_mlps_batched_params(mlps)
        init_batched_params = [p.clone().detach() for p in batched_params]
        optimizer = Adam(batched_params, lr=lr)
        
        for step in range(num_steps):
            selected = random_point_sampling(coords_and_labels, num_points_fitting)
            selected_coords = selected[:, :, :3]
            selected_labels = selected[:, :, 3]
            pred = mlp_batched_forward(batched_params, selected_coords)
            loss = F.binary_cross_entropy_with_logits(pred, selected_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save
        for j in range(bs):
            file_idx = file_indices[j]
            pcd = pcds[j]
            class_id = class_ids[j]
            
            flattened_params = torch.cat([p[j].view(-1) for p in batched_params], dim=0)
            flattened_init = torch.cat([p[j].view(-1) for p in init_batched_params], dim=0)
            
            h5_path = out_root / split / f"{file_idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("pcd", data=pcd.detach().cpu().numpy())
                f.create_dataset("params", data=flattened_params.detach().cpu().numpy())
                f.create_dataset("init_params", data=flattened_init.detach().cpu().numpy())
                f.create_dataset("class_id", data=class_id.detach().cpu().numpy())
        
        print(f"    Batch {batch_start//num_parallel + 1}: saved {bs} files, loss={loss.item():.4f}")
    
    # Verify
    total = len(list((out_root / split).glob("*.h5")))
    print(f"  {split}: {total} total files now.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="cfg/inrs_dataset_with_init.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    out_root = Path(cfg["out_root"])
    pcd_root = Path(cfg["pcd_root"])
    
    for split in cfg["splits"]:
        # Count expected files from source
        src_dir = pcd_root / split
        expected = len(list(src_dir.glob("*.h5")))
        
        missing = find_missing(out_root, split, expected)
        print(f"\n{split}: expected={expected}, missing={len(missing)}")
        
        if missing:
            print(f"  Missing indices (first 20): {missing[:20]}")
            fit_and_save(cfg, split, missing)
        else:
            print(f"  All files present.")


if __name__ == "__main__":
    main()