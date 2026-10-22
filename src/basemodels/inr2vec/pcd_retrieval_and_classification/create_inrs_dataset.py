"""
INR Dataset Creator with Diff Init Support
Based on inr2vec's pcd_retrieval_and_classification/create_inrs_dataset.py
Modified to support:
  - Random (different) initialization per INR
  - Saving init params alongside trained params
  - Configurable via YAML without hesiod dependency
"""

import sys
import yaml
import argparse
from math import ceil
from pathlib import Path
from typing import Callable, List

import h5py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# pycarus imports
from pycarus.datasets.ply import PlyDataset
from pycarus.geometry.pcd import compute_udf_from_pcd, farthest_point_sampling
from pycarus.geometry.pcd import random_point_sampling, shuffle_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.transforms.pcd import JitterPcd, NormalizePcdIntoUnitSphere, RandomScalePcd
from pycarus.utils import progress_bar

# Assumes utils.py is in parent directory
sys.path.append("..")
from utils import get_mlps_batched_params, mlp_batched_forward


class InrsDatasetCreator:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dset_name = cfg["dset_name"]
        self.pcd_root = Path(cfg["pcd_root"])
        self.splits = cfg["splits"]
        self.num_points_pcd = cfg["num_points_pcd"]

        self.num_queries_on_surface = cfg["num_queries_on_surface"]
        self.stds = cfg["stds"]
        self.num_points_per_std = cfg["num_points_per_std"]

        self.num_required_train_shapes = cfg["num_required_train_shapes"]
        dset = self.get_dataset("train")
        num_train_shapes = len(dset)
        self.num_augmentations = ceil(self.num_required_train_shapes / num_train_shapes) - 1

        self.num_points_fitting = cfg["num_points_fitting"]
        self.num_parallel_mlps = cfg["num_parallel_mlps"]
        self.hdim = cfg["mlp"]["hidden_dim"]
        self.num_hidden_layers = cfg["mlp"]["num_hidden_layers"]

        # Init mode: "shared" or "diff"
        self.init_mode = cfg.get("init_mode", "shared")
        if self.init_mode == "shared":
            self.mlp_init_path = Path(cfg["mlp"]["init_path"])
        
        self.num_steps = cfg["num_steps"]
        self.lr = float(cfg["lr"])

        self.out_root = Path(cfg["out_root"])
        self.out_root.mkdir(parents=True, exist_ok=True)

    def build_mlp(self) -> SIREN:
        mlp = SIREN(
            input_dim=3,
            hidden_dim=self.hdim,
            num_hidden_layers=self.num_hidden_layers,
            out_dim=1,
        )

        if self.init_mode == "shared":
            mlp.load_state_dict(torch.load(self.mlp_init_path))
        # else: use default SIREN random init (diff init)

        return mlp

    def get_dataset(self, split: str, transforms: List[Callable] = []) -> Dataset:
        if self.dset_name in ["shapenet10", "scannet10"]:
            dset = PlyDataset(self.pcd_root, split, transforms)
        else:
            raise ValueError(f"Unknown dataset: {self.dset_name}")
        return dset

    def create_dataset(self) -> None:
        for split in self.splits:
            global_idx = 0

            augs = [False]
            if "train" in split:
                augs += [True] * self.num_augmentations

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

                loader = DataLoader(
                    dset,
                    batch_size=self.num_parallel_mlps,
                    shuffle=False,
                    num_workers=8,
                )

                desc = f"Fitting {split} set ({aug_idx + 1}/{len(augs)})"
                for batch in progress_bar(loader, desc, 80):
                    pcds, class_ids = batch

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

                    # Save init params before training
                    init_batched_params = [p.clone().detach() for p in batched_params]

                    optimizer = Adam(batched_params, lr=self.lr)

                    for _ in progress_bar(range(self.num_steps)):
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
                    print(loss.item())
                    for idx in range(bs):
                        pcd = pcds[idx]
                        class_id = class_ids[idx]

                        # Trained params
                        flattened_params = [p[idx].view(-1) for p in batched_params]
                        flattened_params = torch.cat(flattened_params, dim=0)

                        # Init params
                        flattened_init = [p[idx].view(-1) for p in init_batched_params]
                        flattened_init = torch.cat(flattened_init, dim=0)

                        h5_path = self.out_root / split / f"{global_idx}.h5"
                        h5_path.parent.mkdir(parents=True, exist_ok=True)

                        with h5py.File(h5_path, "w") as f:
                            f.create_dataset("pcd", data=pcd.detach().cpu().numpy())
                            f.create_dataset("params", data=flattened_params.detach().cpu().numpy())
                            f.create_dataset("init_params", data=flattened_init.detach().cpu().numpy())
                            f.create_dataset("class_id", data=class_id.detach().cpu().numpy())

                        global_idx += 1

            print(f"Finished {split}: {global_idx} samples saved to {self.out_root / split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="cfg/inrs_dataset_diffinit.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    creator = InrsDatasetCreator(cfg)
    creator.create_dataset()


if __name__ == "__main__":
    main()