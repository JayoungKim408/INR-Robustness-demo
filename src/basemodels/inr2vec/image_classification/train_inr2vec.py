import sys
sys.path.append("..")
import os
import logging
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from basemodels.inr2vec.models.encoder import Encoder
from basemodels.inr2vec.models.idecoder import ImplicitDecoder
from basemodels.utils.nfn.siren_utils import get_mgrid, IMG_shapes
from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset
from utils.wandb_manager import WandbManager

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between predicted and target images."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    # Assuming pixel values in [-1, 1], max value is 2
    return 10 * np.log10(4.0 / mse)


def flatten_mlp_params(sd: Dict[str, Any]) -> Tensor:
    """Flatten state dict parameters to a single tensor."""
    all_params = []
    for k in sd:
        all_params.append(sd[k].view(-1))
    all_params = torch.cat(all_params, dim=-1)
    return all_params


def get_mlp_params_as_matrix_from_wsfeat(weights: Tuple, biases: Tuple, hidden_dim: int) -> Tensor:
    """Convert weights/biases tuple format to matrix format for encoder.

    Skip first and last layers, concatenate middle layer weights and biases.
    """
    # weights: tuple of (1, out, in) tensors
    # biases: tuple of (1, out) tensors
    num_layers = len(weights)

    if num_layers <= 2:
        params = []
        for w, b in zip(weights, biases):
            params.append(w.squeeze(0).reshape(-1))  # flatten weight
            params.append(b.squeeze(0).reshape(-1))  # flatten bias
        params = torch.cat(params)
        return params.reshape(-1, hidden_dim)

    params = []
    for i in range(1, num_layers - 1):
        w = weights[i].squeeze(0)  # (out, in)
        b = biases[i].squeeze(0)   # (out,)
        layer_params = torch.cat([w, b.unsqueeze(-1)], dim=-1)  # (out, in+1)
        params.append(layer_params)

    if len(params) == 0:
        params = []
        for w, b in zip(weights, biases):
            params.append(w.squeeze(0).reshape(-1))
            params.append(b.squeeze(0).reshape(-1))
        params = torch.cat(params)
        return params.reshape(-1, hidden_dim)

    params = torch.cat(params, dim=0)  # (total_neurons, in+1)
    return params


class ImageInrDataset(Dataset):
    """Dataset for 2D image INRs with original images."""

    def __init__(
        self,
        siren_path: str,
        siren_prefix: str,
        data_path: str,
        split: str = "train",
        split_points: Tuple[int, int] = None,
        hidden_dim: int = 32,
        inr_type: str = "siren",
    ) -> None:
        super().__init__()

        self.base_dataset = get_SirenAdnOriginalDataset(
            siren_path=siren_path,
            siren_prefix=siren_prefix,
            data_path=data_path,
            split=split,
            split_points=split_points,
        )
        self.hidden_dim = hidden_dim
        self.data_type = self.base_dataset.data_type

        # Get image shape and create coordinate grid
        self.img_shape = IMG_shapes[self.data_type]
        self.sidelength = self.img_shape[0]
        self.coords = get_mgrid(self.sidelength, dim=2)  # (H*W, 2)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        params, init_params, img, label = self.base_dataset[index]
        weights, biases = params

        # Convert to matrix format for encoder
        matrix = get_mlp_params_as_matrix_from_wsfeat(weights, biases, self.hidden_dim)

        # Image is already normalized, reshape to (H*W, C)
        img_flat = img.permute(1, 2, 0).reshape(-1, img.shape[0])  # (H*W, C)

        return matrix, init_params, self.coords.clone(), img_flat, label


class Inr2vecTrainer:
    def __init__(self, config: Dict[str, Any], wandb_manager: Optional[WandbManager] = None) -> None:
        self.config = config
        self.wandb_manager = wandb_manager

        siren_path = config["siren_path"]
        siren_prefix = config.get("siren_prefix", "shared_init")
        data_path = config["data_path"]
        split_points = config.get("split_points")

        # split_points_map = {"mnist": [55000, 60000], "fashion": [55000, 60000], "cifar": [45000, 50000]}
        # split_points = split_points_map.get(config['dataset_name'], [55000, 60000])

        mlp_hdim = config["mlp"]["hidden_dim"]

        # Create datasets
        train_dset = ImageInrDataset(
            siren_path, siren_prefix, data_path,
            split="train", split_points=split_points, hidden_dim=mlp_hdim,
        )
        if config.get("extra_aug", False):
            aug = []
            for i in range(1, config.get("n_extra_aug", 0)):
                aug.append(ImageInrDataset(
                    siren_path, f"{siren_prefix}_{i}", data_path,
                    split="train", split_points=split_points, hidden_dim=mlp_hdim,
                )
            )
            
            train_dset = ConcatDataset([train_dset] + aug)
        train_bs = config.get("train_bs", 64)
        self.train_loader = DataLoader(
            train_dset, batch_size=train_bs, num_workers=4, shuffle=True
        )

        val_dset = ImageInrDataset(
            siren_path, siren_prefix, data_path,
            split="val", split_points=split_points, hidden_dim=mlp_hdim,
        )
        val_bs = config.get("val_bs", 64)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=4)
        print(f"Dataset sizes: train={len(train_dset)}, val={len(val_dset)}, batch_size={train_bs}, siren_prefix: {siren_prefix}, extra_aug: {config.get('extra_aug', False)}, n_extra_aug: {config.get('n_extra_aug', False)}")
        
        self.data_type = train_dset.data_type
        self.img_channels = IMG_shapes[self.data_type][2]

        # Get matrix dimension from first sample
        # atrix, init_params, self.coords.clone(), img_flat, label
        sample_matrix, _, _, _, _ = train_dset[0]
        matrix_feat_dim = sample_matrix.shape[-1]

        # Encoder configuration
        encoder_cfg = config.get("encoder", {})
        encoder = Encoder(
            matrix_feat_dim,
            encoder_cfg.get("hidden_dims", [128, 256, 512]),
            encoder_cfg.get("embedding_dim", 256),
        )
        self.encoder = encoder.cuda()

        # Decoder configuration - 2D coordinates
        decoder_cfg = config.get("decoder", {})
        decoder = ImplicitDecoder(
            encoder_cfg.get("embedding_dim", 256),
            in_dim=decoder_cfg.get("input_dim", 2),  # 2D coordinates
            hidden_dim=decoder_cfg.get("hidden_dim", 256),
            num_hidden_layes_before_skip=decoder_cfg.get("num_hidden_layers_before_skip", 3),
            num_hidden_layes_after_skip=decoder_cfg.get("num_hidden_layers_after_skip", 3),
            out_dim=self.img_channels,  # 1 for grayscale, 3 for RGB
        )
        self.decoder = decoder.cuda()
        print(encoder)
        print(decoder)  
        # Optimizer
        lr = config.get("lr", 1e-4)
        wd = config.get("wd", 1e-5)
        params = list(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        self.optimizer = AdamW(params, lr, weight_decay=wd)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Checkpoint paths
        self.ckpts_path = Path(config.get("out_dir", "outputs/image_inr2vec"))
        self.ckpts_path.mkdir(parents=True, exist_ok=True)

        # Restore from checkpoint if exists
        if (self.ckpts_path / "last.pt").exists():
            self.restore_from_last_ckpt()

    def logfn(self, values: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb via WandbManager."""
        if step is None:
            step = self.global_step
        if self.wandb_manager is not None:
            self.wandb_manager.log(values, step=step)
        elif wandb.run is not None:
            wandb.log(values, step=step, commit=False)

    def train(self) -> None:
        num_epochs = self.config.get("num_epochs", 100)
        start_epoch = self.epoch
        log_interval = self.config.get("log_interval", 50)

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            self.encoder.train()
            self.decoder.train()

            epoch_losses = []
            epoch_psnrs = []

            for batch in self.train_loader:
                matrices, init_matrices, coords, img_labels, class_labels = batch
                matrices = matrices.cuda().float()
                # init_matrices = init_matrices.cuda().float()
                coords = coords.cuda().float()
                img_labels = img_labels.cuda().float()

                # Random point sampling for fitting (None = use all pixels)
                batch_size = coords.shape[0]
                num_points = coords.shape[1]

                selected_coords = coords
                selected_labels = img_labels

                # Forward pass
                embeddings = self.encoder(matrices)
                pred = self.decoder(embeddings, selected_coords)

                # For multi-channel output, reshape pred
                if self.img_channels > 1:
                    pred = pred.view(batch_size, -1, self.img_channels)
                else:
                    pred = pred.unsqueeze(-1)

                # MSE loss for pixel reconstruction
                loss = F.mse_loss(pred, selected_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

                # Compute PSNR
                with torch.no_grad():
                    psnr = compute_psnr(pred, selected_labels)
                    epoch_psnrs.append(psnr)

                # Log every N steps
                if self.global_step % log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logfn({
                        "train/loss": loss.item(),
                        "train/psnr": psnr,
                        "train/lr": current_lr,
                    })

                self.global_step += 1

            # Epoch-level logging
            avg_loss = np.mean(epoch_losses)
            avg_psnr = np.mean(epoch_psnrs)
            self.logfn({
                "train/epoch_loss": avg_loss,
                "train/epoch_psnr": avg_psnr,
                "train/epoch": epoch,
            })
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f}")

            # Validation
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_metrics = self.validate()
                self.logfn({
                    "val/loss": val_metrics["loss"],
                    "val/psnr": val_metrics["psnr"],
                })
                print(f"  Val Loss: {val_metrics['loss']:.6f}, Val PSNR: {val_metrics['psnr']:.2f}")

                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.save_ckpt(best=True)

            # Log reconstructed images every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.log_reconstructed_images(num_samples=8)

            self.save_ckpt()

    @torch.no_grad()
    def log_reconstructed_images(self, num_samples: int = 8) -> None:
        """Log reconstructed images to wandb for visualization."""
        if self.wandb_manager is None and wandb.run is None:
            return

        self.encoder.eval()
        self.decoder.eval()

        # Get a batch from validation loader
        batch = next(iter(self.val_loader))
        matrices, init_matrices, coords, img_labels, class_labels = batch

        # Limit to num_samples
        matrices = matrices[:num_samples].cuda().float()
        # init_matrices = init_matrices[:num_samples].cuda().float()  
        coords = coords[:num_samples].cuda().float()
        img_labels = img_labels[:num_samples].cuda().float()
        class_labels = class_labels[:num_samples]

        batch_size = matrices.shape[0]

        # Forward pass
        embeddings = self.encoder(matrices)
        pred = self.decoder(embeddings, coords)

        # Reshape predictions
        if self.img_channels > 1:
            pred = pred.view(batch_size, -1, self.img_channels)
        else:
            pred = pred.unsqueeze(-1)

        # Get image shape
        img_shape = IMG_shapes[self.data_type]
        H, W, C = img_shape

        # Reshape to image format: (B, H*W, C) -> (B, C, H, W)
        pred_imgs = pred.view(batch_size, H, W, C).permute(0, 3, 1, 2)
        gt_imgs = img_labels.view(batch_size, H, W, C).permute(0, 3, 1, 2)

        # Normalize from [-1, 1] to [0, 1] for visualization
        pred_imgs = (pred_imgs + 1) / 2
        gt_imgs = (gt_imgs + 1) / 2

        # Clamp to valid range
        pred_imgs = pred_imgs.clamp(0, 1)
        gt_imgs = gt_imgs.clamp(0, 1)

        # Create wandb images
        wandb_images = []
        for i in range(batch_size):
            # Convert to numpy (C, H, W) -> (H, W, C)
            pred_np = pred_imgs[i].cpu().permute(1, 2, 0).numpy()
            gt_np = gt_imgs[i].cpu().permute(1, 2, 0).numpy()

            # For grayscale, squeeze the last dimension
            if C == 1:
                pred_np = pred_np.squeeze(-1)
                gt_np = gt_np.squeeze(-1)

            # Create side-by-side comparison
            if C == 1:
                combined = np.concatenate([gt_np, pred_np], axis=1)
            else:
                combined = np.concatenate([gt_np, pred_np], axis=1)

            wandb_images.append(wandb.Image(
                combined,
                caption=f"Label {class_labels[i].item()} | GT (left) vs Recon (right)"
            ))

        # Log images
        log_dict = {"val/reconstructions": wandb_images}
        if self.wandb_manager is not None:
            self.wandb_manager.log(log_dict, step=self.global_step)
        elif wandb.run is not None:
            wandb.log(log_dict, step=self.global_step, commit=False)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.encoder.eval()
        self.decoder.eval()

        losses = []
        psnrs = []

        for batch in self.val_loader:
            matrices, init_matrices, coords, img_labels, class_labels = batch
            matrices = matrices.cuda().float()
            coords = coords.cuda().float()
            img_labels = img_labels.cuda().float()

            batch_size = coords.shape[0]

            embeddings = self.encoder(matrices)
            pred = self.decoder(embeddings, coords)

            if self.img_channels > 1:
                pred = pred.view(batch_size, -1, self.img_channels)
            else:
                pred = pred.unsqueeze(-1)

            loss = F.mse_loss(pred, img_labels)
            losses.append(loss.item())

            psnr = compute_psnr(pred, img_labels)
            psnrs.append(psnr)

        return {
            "loss": np.mean(losses),
            "psnr": np.mean(psnrs),
        }

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        torch.save(ckpt, self.ckpts_path / "last.pt")

        if best:
            torch.save(ckpt, self.ckpts_path / "best.pt")

    def restore_from_last_ckpt(self) -> None:
        ckpt_path = self.ckpts_path / "last.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = ckpt.get("global_step", self.epoch * len(self.train_loader))
            self.best_loss = ckpt["best_loss"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])

            print(f"Restored from epoch {self.epoch - 1}")


def main(config: Dict[str, Any] = None) -> None:
    if config is None:
        # Default configuration
        config = {
            "siren_path": "/path/to/siren/data",
            "siren_prefix": "randinit_test",
            "data_path": "/path/to/image/data",
            "split_points": [55000, 60000],
            # "mlp_hidden_dim": 32,
            "train_bs": 64,
            "val_bs": 64,
            "encoder": {
                "hidden_dims": [128, 256, 512],
                "embedding_dim": 256,
            },
            "decoder": {
                "hidden_dim": 256,
                "num_hidden_layers_before_skip": 3,
                "num_hidden_layers_after_skip": 3,
            },
            "lr": 1e-4,
            "wd": 1e-5,
            "num_epochs": 100,
            "out_dir": "outputs/image_inr2vec",
        }

    # Initialize wandb if not already initialized
    if wandb.run is None:
        wandb.init(
            project="inr2vec-image",
            config=config,
        )

    trainer = Inr2vecTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
