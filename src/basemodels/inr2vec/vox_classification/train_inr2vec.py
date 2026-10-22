import sys

sys.path.append("..")

import logging
import os
from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import wandb
import yaml
from einops import rearrange
from pycarus.geometry.pcd import random_point_sampling, voxelize_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.metrics.chamfer_distance import chamfer_t
from pycarus.metrics.f_score import f_score
from pycarus.utils import progress_bar
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from utils import focal_loss, get_mlp_params_as_matrix

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


class InrDataset(Dataset):
    def __init__(self, inrs_root, split, sample_sd, vox_res):
        super().__init__()
        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_sd = sample_sd
        self.vox_res = vox_res

    def __len__(self):
        return len(self.mlps_paths)

    def __getitem__(self, index):
        with h5py.File(self.mlps_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            params = np.array(f.get("params"))
            params = torch.from_numpy(params).float()
            matrix = get_mlp_params_as_matrix(params, self.sample_sd)
        vgrid, centroids = voxelize_pcd(pcd, self.vox_res, -1, 1)
        return vgrid, centroids, matrix


class Inr2vecTrainer:
    def __init__(self, cfg):
        inrs_root = cfg["inrs_root"]
        self.num_points_fitting = cfg["num_points_fitting"]

        mlp_hdim = cfg["mlp"]["hidden_dim"]
        num_hidden_layers = cfg["mlp"]["num_hidden_layers"]
        mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
        sample_sd = mlp.state_dict()

        vox_res = cfg["vox_res"]

        train_dset = InrDataset(inrs_root, cfg["train_split"], sample_sd, vox_res)
        self.train_loader = DataLoader(train_dset, batch_size=cfg["train_bs"], num_workers=8, shuffle=True)

        val_dset = InrDataset(inrs_root, cfg["val_split"], sample_sd, vox_res)
        self.val_loader = DataLoader(val_dset, batch_size=cfg["val_bs"], num_workers=8)
        self.val_loader_shuffled = DataLoader(val_dset, batch_size=cfg["val_bs"], num_workers=8, shuffle=True)

        enc_cfg = cfg["encoder"]
        self.encoder = Encoder(mlp_hdim, enc_cfg["hidden_dims"], enc_cfg["embedding_dim"]).cuda()

        dec_cfg = cfg["decoder"]
        self.decoder = ImplicitDecoder(
            enc_cfg["embedding_dim"],
            dec_cfg["input_dim"],
            dec_cfg["hidden_dim"],
            dec_cfg["num_hidden_layers_before_skip"],
            dec_cfg["num_hidden_layers_after_skip"],
            dec_cfg["out_dim"],
        ).cuda()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = AdamW(params, cfg["lr"], weight_decay=cfg["wd"])

        self.num_epochs = cfg["num_epochs"]
        self.epoch = 0
        self.global_step = 0
        self.best_chamfer = float("inf")

        self.out_dir = Path(cfg.get("out_dir", "outputs"))
        self.ckpts_path = self.out_dir / "ckpts"
        self.all_ckpts_path = self.out_dir / "all_ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.all_ckpts_path.mkdir(parents=True, exist_ok=True)

    def logfn(self, values):
        wandb.log(values, step=self.global_step, commit=False)

    def train(self):
        import time
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            # for batch in progress_bar(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
            t0 = time.time()

            for i, batch in enumerate(self.train_loader):
                vgrids, centroids, matrices = batch
                vgrids = vgrids.cuda()
                centroids = centroids.cuda()
                matrices = matrices.cuda()

                coords = rearrange(centroids, "b r1 r2 r3 d -> b (r1 r2 r3) d")
                labels = rearrange(vgrids, "b r1 r2 r3 -> b (r1 r2 r3)")

                coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1)
                selected_c_and_l = random_point_sampling(coords_and_labels, self.num_points_fitting)

                selected_coords = selected_c_and_l[:, :, :3]
                selected_labels = selected_c_and_l[:, :, 3]

                embeddings = self.encoder(matrices)
                pred = self.decoder(embeddings, selected_coords)
                loss = focal_loss(pred, selected_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                
                if (i % 1000 == 0) and (i > 1):
                    elapsed = time.time() - t0
                    eta = elapsed / i * (len(self.train_loader) - i)
                    print(f"Epoch {self.epoch}/{self.num_epochs} - Step {i}/{len(self.train_loader)} - Loss: {loss.item():.4f} - ETA: {eta/60:.1f}min")                

                self.global_step += 1

            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                self.val("train")
                self.val("val")
                self.plot("train")
                self.plot("val")

            if epoch % 10 == 0:
                self.save_ckpt(all=True)
            self.save_ckpt()

    @torch.no_grad()
    def val(self, split):
        loader = self.train_loader if split == "train" else self.val_loader
        self.encoder.eval()
        self.decoder.eval()

        cdts, fscores = [], []
        idx = 0
        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            vgrids, centroids, matrices = batch
            vgrids = vgrids.cuda()
            centroids = centroids.cuda()
            matrices = matrices.cuda()
            bs = vgrids.shape[0]
            embeddings = self.encoder(matrices)

            for i in range(bs):
                pcd_gt = centroids[i][vgrids[i] == 1]
                centr = centroids[i].unsqueeze(0)
                centr = rearrange(centr, "b r1 r2 r3 d -> b (r1 r2 r3) d")
                emb = embeddings[i].unsqueeze(0)
                vgrid_pred = torch.sigmoid(self.decoder(emb, centr).squeeze(0))
                pcd_pred = centr[0][vgrid_pred > 0.4]
                cdts.append(float(chamfer_t(pcd_pred, pcd_gt)))
                fscores.append(float(f_score(pcd_pred, pcd_gt, threshold=0.01)[0]))

            if idx > 99:
                break
            idx += 1

        self.logfn({f"{split}/cdt": sum(cdts) / len(cdts)})
        self.logfn({f"{split}/fscore": sum(fscores) / len(fscores)})

        if split == "val" and sum(cdts) / len(cdts) < self.best_chamfer:
            self.best_chamfer = sum(cdts) / len(cdts)
            self.save_ckpt(best=True)

    @torch.no_grad()
    def plot(self, split):
        loader = self.train_loader if split == "train" else self.val_loader_shuffled
        self.encoder.eval()
        self.decoder.eval()

        batch = next(iter(loader))
        vgrids, centroids, matrices = batch
        vgrids = vgrids.cuda()
        centroids = centroids.cuda()
        matrices = matrices.cuda()
        embeddings = self.encoder(matrices)

        for i in range(vgrids.shape[0]):
            pcd_gt = centroids[i][vgrids[i] == 1]
            centr = centroids[i].unsqueeze(0)
            centr = rearrange(centr, "b r1 r2 r3 d -> b (r1 r2 r3) d")
            emb = embeddings[i].unsqueeze(0)
            vgrid_pred = torch.sigmoid(self.decoder(emb, centr).squeeze(0))
            pcd_pred = centr[0][vgrid_pred > 0.4]
            pcd_pred = random_point_sampling(pcd_pred, 2048)

            gt_wo3d = wandb.Object3D(pcd_gt.cpu().detach().numpy())
            pred_wo3d = wandb.Object3D(pcd_pred.cpu().detach().numpy())
            self.logfn({f"{split}/voxel_{i}": [gt_wo3d, pred_wo3d]})

    def save_ckpt(self, best=False, all=False):
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_chamfer": self.best_chamfer,
        }
        if all:
            torch.save(ckpt, self.all_ckpts_path / f"{self.epoch}.pt")
        else:
            for p in self.ckpts_path.glob("*.pt"):
                if "best" not in p.name:
                    p.unlink()
            torch.save(ckpt, self.ckpts_path / f"{self.epoch}.pt")
        if best:
            torch.save(ckpt, self.ckpts_path / "best.pt")

    def restore_from_last_ckpt(self):
        ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
        assert len(ckpt_paths) == 1
        ckpt = torch.load(ckpt_paths[0])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = self.epoch * len(self.train_loader)
        self.best_chamfer = ckpt["best_chamfer"]
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "vox_classification/cfg/inr2vec.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project="inr2vec",
        name=cfg.get("run_name", "inr2vec"),
        dir=cfg.get("out_dir", "outputs"),
        config=cfg,
    )

    trainer = Inr2vecTrainer(cfg)
    trainer.train()