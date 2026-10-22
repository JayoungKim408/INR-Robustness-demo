"""
INR Dataset Creator with Diff Init Support (DDP version)
Based on inr2vec's pcd_retrieval_and_classification/create_inrs_dataset.py
Modified to support:
  - Random (different) initialization per INR
  - Saving init params alongside trained params
  - Configurable via YAML without hesiod dependency
  - Multi-GPU via DDP (data-parallel dataset creation)
"""

import os
import sys
# sys.path.append("..")
sys.path.insert(0, "/workspace/INR-Robustness/src")  # Ensure src is in path for imports

import yaml
import argparse
from math import ceil
from pathlib import Path
from typing import Callable, List

import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# pycarus imports
from pcd import compute_udf_from_pcd, farthest_point_sampling, random_point_sampling, shuffle_pcd
from transform_pcd import JitterPcd, NormalizePcdIntoUnitSphere, RandomScalePcd
from basemodels.utils.nfn.siren_utils import Siren
from basemodels.inr2vec.utils import get_mlps_batched_params, mlp_batched_forward, progress_bar


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


class H5PcdDataset(Dataset):
    """Read point clouds and class_ids from existing h5 files."""
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
        # Also return original index for deterministic file naming
        original_idx = int(self.h5_paths[idx].stem)
        for t in self.transforms:
            pcd = t(pcd)
        return pcd, class_id, original_idx


class InrsDatasetCreator:
    def __init__(self, cfg, rank, world_size, local_rank) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        self.pcd_root = Path(cfg["pcd_root"])
        self.splits = cfg["splits"]
        self.num_points_pcd = cfg["num_points_pcd"]

        self.num_queries_on_surface = cfg["num_queries_on_surface"]
        self.stds = cfg["stds"]
        self.num_points_per_std = cfg["num_points_per_std"]

        self.num_required_train_shapes = cfg.get("num_required_train_shapes", 0)
        dset = self.get_dataset("train")
        num_train_shapes = len(dset)
        if self.num_required_train_shapes > 0:
            self.num_augmentations = ceil(self.num_required_train_shapes / num_train_shapes) - 1
        else:
            self.num_augmentations = 0

        self.num_points_fitting = cfg["num_points_fitting"]
        self.num_parallel_mlps = cfg["num_parallel_mlps"]
        self.hdim = cfg["mlp"]["hidden_dim"]
        self.num_hidden_layers = cfg["mlp"]["num_hidden_layers"]

        self.init_mode = cfg.get("init_mode", "shared")
        if self.init_mode == "shared":
            self.mlp_init_path = Path(cfg["mlp"]["init_path"])

        self.num_steps = cfg["num_steps"]
        self.lr = float(cfg["lr"])

        self.out_root = Path(cfg["out_root"])
        if self.rank == 0:
            self.out_root.mkdir(parents=True, exist_ok=True)
        dist.barrier()

    def build_mlp(self) -> nn.Module:
        mlp = Siren(
            in_features=3,
            out_features=1,
            hidden_features=self.hdim,
            hidden_layers=self.num_hidden_layers,
            outermost_linear=True,
            first_omega_0=30,
            hidden_omega_0=30,
            is_first=self.cfg.get("is_first", True),
        )

        if self.cfg.get("bias_zero_init", False):
            for name, param in mlp.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)

        if self.init_mode == "shared":
            mlp.load_state_dict(torch.load(self.mlp_init_path, map_location="cpu"))

        return mlp

    def get_dataset(self, split: str, transforms: List[Callable] = []) -> Dataset:
        return H5PcdDataset(self.pcd_root, split, transforms)

    def create_dataset(self) -> None:
        for split in self.splits:
            augs = [False]
            if "train" in split:
                augs += [True] * self.num_augmentations

            global_offset = 0

            for aug_idx, aug in enumerate(augs):
                if aug:
                    transforms = [
                        RandomScalePcd(2 / 3, 3 / 2),
                        JitterPcd(sigma=0.01, clip=0.05),
                        NormalizePcdIntoUnitSphere(),
                    ]
                else:
                    transforms = [NormalizePcdIntoUnitSphere()]

                dset = self.get_dataset(split, transforms)

                sampler = DistributedSampler(
                    dset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )

                loader = DataLoader(
                    dset,
                    batch_size=self.num_parallel_mlps,
                    shuffle=False,
                    num_workers=8,
                    sampler=sampler,
                    drop_last=False,
                )

                desc = f"[Rank {self.rank}] Fitting {split} set ({aug_idx + 1}/{len(augs)})"
                for batch in progress_bar(loader, desc, 80):
                    pcds, class_ids, original_indices = batch

                    bs = pcds.shape[0]
                    pcds = pcds.cuda()

                    if pcds.shape[1] != self.num_points_pcd:
                        pcds = farthest_point_sampling(pcds, self.num_points_pcd)

                    coords = []
                    labels = []
                    for idx in range(bs):
                        pcd_coords, pcd_labels = compute_udf_from_pcd(
                            pcds[idx],
                            self.num_queries_on_surface,
                            self.stds,
                            self.num_points_per_std,
                            coords_range=(-1, 1),
                            convert_to_bce_labels=True,
                        )
                        coords.append(pcd_coords)
                        labels.append(pcd_labels)

                    coords = torch.stack(coords, dim=0)
                    labels = torch.stack(labels, dim=0)

                    coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1).cuda()
                    coords_and_labels = shuffle_pcd(coords_and_labels)

                    mlps = [self.build_mlp().cuda() for _ in range(bs)]
                    batched_params = get_mlps_batched_params(mlps)

                    init_batched_params = [p.clone().detach() for p in batched_params]

                    optimizer = Adam(batched_params, lr=self.lr)

                    # for _ in progress_bar(range(self.num_steps)):
                    for step in range(self.num_steps):
                        selected_c_and_l = random_point_sampling(
                            coords_and_labels,
                            self.num_points_fitting,
                        )

                        selected_coords = selected_c_and_l[:, :, :3]
                        selected_labels = selected_c_and_l[:, :, 3]

                        pred = mlp_batched_forward(batched_params, selected_coords)
                        loss = F.binary_cross_entropy_with_logits(pred, selected_labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if self.rank == 0 and step % 100 == 0 and step > 0:
                            print(f"  Step {step}: loss={loss.item():.4f}")

                    for idx in range(bs):
                        pcd = pcds[idx]
                        class_id = class_ids[idx]
                        # Use original index + augmentation offset for deterministic naming
                        file_idx = global_offset + original_indices[idx].item()

                        flattened_params = [p[idx].view(-1) for p in batched_params]
                        flattened_params = torch.cat(flattened_params, dim=0)

                        flattened_init = [p[idx].view(-1) for p in init_batched_params]
                        flattened_init = torch.cat(flattened_init, dim=0)

                        h5_path = self.out_root / split / f"{file_idx}.h5"
                        h5_path.parent.mkdir(parents=True, exist_ok=True)

                        with h5py.File(h5_path, "w") as f:
                            f.create_dataset("pcd", data=pcd.detach().cpu().numpy())
                            f.create_dataset("params", data=flattened_params.detach().cpu().numpy())
                            f.create_dataset("init_params", data=flattened_init.detach().cpu().numpy())
                            f.create_dataset("class_id", data=class_id.detach().cpu().numpy())

                    # if self.rank == 0 :
                    #     print(f"{desc} batch {batch_idx}/{len(loader)}")

                global_offset += len(dset)
                dist.barrier()

            if self.rank == 0:
                # Count total files saved
                total = len(list((self.out_root / split).glob("*.h5")))
                print(f"Finished {split}: {total} samples saved to {self.out_root / split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="cfg/inrs_dataset_with_init.yaml")
    args = parser.parse_args()

    local_rank, rank, world_size = setup_ddp()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    creator = InrsDatasetCreator(cfg, rank, world_size, local_rank)
    creator.create_dataset()

    cleanup_ddp()


if __name__ == "__main__":
    main()