import sys
sys.path.append("..")

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
# from pycarus.geometry.pcd import compute_udf_from_pcd, sample_pcds_from_udfs
# from pycarus.learning.models.siren import SIREN
# from pycarus.metrics.chamfer_distance import chamfer_t
# from pycarus.metrics.f_score import f_score
# from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from basemodels.inr2vec.models.encoder import Encoder
from basemodels.inr2vec.models.idecoder import ImplicitDecoder
from basemodels.inr2vec.utils import get_mlp_params_as_matrix
from basemodels.inr2vec.pcd_retrieval_and_classification.siren_utils import SIREN
from basemodels.inr2vec.pcd_retrieval_and_classification.geometry_pcd import compute_udf_from_pcd, sample_pcds_from_udfs
from basemodels.inr2vec.pcd_retrieval_and_classification.metrics import chamfer_t, f_score
from utils.wandb_manager import WandbManager

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


class InrDataset(Dataset):
    def __init__(self, inrs_root, split, siren_prefix, sample_sd):
        super().__init__()
        self.inrs_root = Path(inrs_root) / siren_prefix / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_sd = sample_sd

    def __len__(self):
        return len(self.mlps_paths)

    def __getitem__(self, index):
        with h5py.File(self.mlps_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            params = np.array(f.get("params"))
            params = torch.from_numpy(params).float()
            matrix = get_mlp_params_as_matrix(params, self.sample_sd)
        return pcd, matrix


class Inr2vecTrainer:
    def __init__(self, config: Dict[str, Any], wandb_manager: Optional[WandbManager] = None):
        self.config = config
        self.wandb_manager = wandb_manager

        inrs_root = config["siren_path"]
        self.num_queries_on_surface = config["num_queries_on_surface"]
        self.stds = config["stds"]
        self.num_points_per_std = config["num_points_per_std"]

        mlp_hdim = config["mlp"]["hidden_dim"]
        num_hidden_layers = config["mlp"]["num_hidden_layers"]
        mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
        sample_sd = mlp.state_dict()

        train_dset = InrDataset(inrs_root, "train", config["siren_prefix"], sample_sd)
        self.train_loader = DataLoader(train_dset, batch_size=config.get("train_bs", 16), num_workers=8, shuffle=True)

        val_dset = InrDataset(inrs_root, "val", config["siren_prefix"], sample_sd)
        self.val_loader = DataLoader(val_dset, batch_size=config.get("val_bs", 16), num_workers=8)

        print(f"Dataset sizes: train={len(train_dset)}, val={len(val_dset)}")

        enc_cfg = config["encoder"]
        self.encoder = Encoder(mlp_hdim, enc_cfg["hidden_dims"], enc_cfg["embedding_dim"]).cuda()

        dec_cfg = config["decoder"]
        self.decoder = ImplicitDecoder(
            enc_cfg["embedding_dim"],
            dec_cfg["input_dim"],
            dec_cfg["hidden_dim"],
            dec_cfg["num_hidden_layers_before_skip"],
            dec_cfg["num_hidden_layers_after_skip"],
            dec_cfg["out_dim"],
        ).cuda()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = AdamW(params, float(config.get("lr", 1e-4)), weight_decay=float(config.get("wd", 1e-2)))

        self.num_epochs = config.get("num_epochs", 300)
        self.epoch = 0
        self.global_step = 0
        self.best_chamfer = float("inf")

        self.ckpts_path = Path(config.get("out_dir", "outputs"))
        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        # self.out_dir.mkdir(parents=True, exist_ok=True)
        # self.ckpts_path = self.out_dir / "ckpts"
        # self.all_ckpts_path = self.out_dir / "all_ckpts"

        # if self.ckpts_path.exists():
        if (self.ckpts_path / "last.pt").exists():
            self.restore_from_last_ckpt()

        # self.all_ckpts_path.mkdir(parents=True, exist_ok=True)

    def logfn(self, values, step=None):
        if step is None:
            step = self.global_step
        if self.wandb_manager is not None:
            self.wandb_manager.log(values, step=step)

    def train(self):
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            t0 = time.time()
            for i, batch in enumerate(self.train_loader):
                pcds, matrices = batch
                matrices = matrices.cuda()
                pcds = pcds.cuda()

                coords = []
                labels = []
                for j in range(pcds.shape[0]):
                    crd, lbl = compute_udf_from_pcd(
                        pcds[j],
                        self.num_queries_on_surface,
                        self.stds,
                        self.num_points_per_std,
                        (-1, 1),
                        convert_to_bce_labels=True,
                    )
                    coords.append(crd)
                    labels.append(lbl)
                coords = torch.stack(coords, dim=0).cuda()
                labels = torch.stack(labels, dim=0).cuda()

                embeddings = self.encoder(matrices)
                pred = self.decoder(embeddings, coords)
                loss = F.binary_cross_entropy_with_logits(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})

                if i % 100 == 0 and i > 0:
                # if i > 0 :
                    elapsed = time.time() - t0
                    eta = elapsed / i * (len(self.train_loader) - i)
                    print(f"Epoch {self.epoch}/{self.num_epochs} - Step {i}/{len(self.train_loader)} - Loss: {loss.item():.4f} - ETA: {eta/60:.1f}min")

                self.global_step += 1

            print(f"Epoch {self.epoch}/{self.num_epochs} - Loss: {loss.item():.4f}")

            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                # self.val("train")
                self.val("val")

            if epoch % 50 == 0:
                self.save_ckpt(all=True)
            self.save_ckpt()

    # @torch.no_grad()
    def val(self, split):
        loader = self.train_loader if split == "train" else self.val_loader
        self.encoder.eval()
        self.decoder.eval()

        cdts, fscores = [], []
        idx = 0
        from tqdm import tqdm 
        for batch in tqdm(loader):
            pcds, matrices = batch
            pcds = pcds.cuda()
            matrices = matrices.cuda()
            bs = pcds.shape[0]
            
            with torch.no_grad():
                embeddings = self.encoder(matrices)

            for j in range(bs):
                emb = embeddings[j].unsqueeze(0)

                def udfs_func(coords, indices):
                    coords = coords.cuda()
                    pred = torch.sigmoid(self.decoder(emb, coords))
                    pred = 1 - pred
                    pred *= 0.1
                    return pred

                pred_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, 2048, 1)
                cdts.append(float(chamfer_t(pred_pcd[0], pcds[j])))
                fscores.append(float(f_score(pred_pcd[0], pcds[j], threshold=0.01)[0]))

            if idx > 99:
                break
            idx += 1

        mean_cdt = sum(cdts) / len(cdts)
        mean_fscore = sum(fscores) / len(fscores)

        self.logfn({f"{split}/cdt": mean_cdt})
        self.logfn({f"{split}/fscore": mean_fscore})
        print(f"  {split} - cdt: {mean_cdt:.6f}, fscore: {mean_fscore:.4f}")

        if split == "val" and mean_cdt < self.best_chamfer:
            self.best_chamfer = mean_cdt
            self.save_ckpt(best=True)

    def save_ckpt(self, best=False, all=False):
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_chamfer": self.best_chamfer,
            "config": self.config,
        }
        # if all:
        #     torch.save(ckpt, self.all_ckpts_path / f"{self.epoch}.pt")
        # else:
        for p in self.ckpts_path.glob("*.pt"):
            if "best" not in p.name:
                p.unlink()
        torch.save(ckpt, self.ckpts_path / f"{self.epoch}.pt")
        if best:
            torch.save(ckpt, self.ckpts_path / "best.pt")

    def restore_from_last_ckpt(self):
        ckpt_path = self.ckpts_path / "last.pt"

        # ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
        # assert len(ckpt_paths) == 1
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.epoch = ckpt["epoch"] + 1
        self.global_step = self.epoch * len(self.train_loader)
        self.best_chamfer = ckpt["best_chamfer"]
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Restored from epoch {self.epoch - 1}")


def main(config: Dict[str, Any], wandb_manager: Optional[WandbManager] = None):
    trainer = Inr2vecTrainer(config, wandb_manager)
    trainer.train()
    return str(trainer.ckpts_path / "best.pt")