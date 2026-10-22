import sys
sys.path.append("..")
import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from basemodels.inr2vec.models.fc_classifier import FcClassifier
from utils.wandb_manager import WandbManager


class AccuracyMetric:
    """Simple accuracy metric without torchmetrics dependency."""
    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = pred.argmax(dim=-1)
        self.correct += (pred_labels == target).sum().item()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.correct / self.total if self.total > 0 else 0.0)

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


class EmbeddingDataset(Dataset):
    """Dataset for loading pre-extracted embeddings."""

    def __init__(self, embeddings_path: str, split: str = "train") -> None:
        super().__init__()

        data = torch.load(embeddings_path, weights_only=False)
        all_embeddings = data["embeddings"]
        all_labels = data["labels"]
        split_points = data["split_points"]

        if split == "train":
            self.embeddings = all_embeddings[:split_points[0]]
            self.labels = all_labels[:split_points[0]]
        elif split == "val":
            self.embeddings = all_embeddings[split_points[0]:split_points[1]]
            self.labels = all_labels[split_points[0]:split_points[1]]
        elif split == "test":
            self.embeddings = all_embeddings[split_points[1]:]
            self.labels = all_labels[split_points[1]:]
        else:  # all
            self.embeddings = all_embeddings
            self.labels = all_labels

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.embeddings[index], self.labels[index].long()


class EmbeddingClassifier:
    def __init__(self, config: Dict[str, Any], wandb_manager: Optional[WandbManager] = None) -> None:
        self.config = config
        self.wandb_manager = wandb_manager

        embeddings_path = config["embeddings_path"]

        # Create datasets
        train_dset = EmbeddingDataset(embeddings_path, "train")
        train_bs = config.get("train_bs", 256)
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=4, shuffle=True)

        val_dset = EmbeddingDataset(embeddings_path, "val")
        val_bs = config.get("val_bs", 256)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=4)

        test_dset = EmbeddingDataset(embeddings_path, "test")
        self.test_loader = DataLoader(test_dset, batch_size=val_bs, num_workers=4)
        print(f'Dataset size: train: {len(train_dset)}, val: {len(val_dset)}, test: {len(test_dset)}')
        
        # Get embedding dimension from dataset
        sample_emb, _ = train_dset[0]
        embed_dim = sample_emb.shape[-1]

        # Classifier configuration
        layers_dim = config.get("layers_dim", [embed_dim, 256, 128])
        if layers_dim[0] != embed_dim:
            layers_dim = [embed_dim] + layers_dim[1:]
            # Update config with corrected layers_dim so checkpoint saves correct value
            config["layers_dim"] = layers_dim

        self.num_classes = config.get("num_classes", 10)
        net = FcClassifier(layers_dim, self.num_classes)
        self.net = net.cuda()
        print(net)
        # Optimizer and scheduler
        lr = config.get("lr", 1e-3)
        wd = config.get("wd", 1e-4)
        self.optimizer = AdamW(self.net.parameters(), lr, weight_decay=wd)

        num_epochs = config.get("num_epochs", 100)
        num_steps = num_epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, lr, total_steps=num_steps)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0
        self.best_model = None

        # Checkpoint paths
        self.ckpts_path = Path(config.get("out_dir", "outputs/image_classifier"))
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

    def interpolate(self, params: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Data augmentation by interpolating embeddings of same class."""
        new_params = []
        new_labels = []

        for i, p1 in enumerate(params):
            same_class = labels == labels[i]
            num_same_class = torch.sum(same_class)

            if num_same_class > 2:
                indices = torch.where(same_class)[0]
                random_order = torch.randperm(len(indices))
                random_idx = indices[random_order][0]
                p2 = params[random_idx]

                random_uniform = torch.rand(len(p2))
                tsh = torch.rand(())
                from2 = random_uniform >= tsh
                p1_copy = p1.clone()
                p1_copy[from2] = p2[from2]
                new_params.append(p1_copy)
                new_labels.append(labels[i])

        if len(new_params) > 0:
            new_params = torch.stack(new_params)
            final_params = torch.cat([params, new_params], dim=0)
            new_labels = torch.stack(new_labels)
            final_labels = torch.cat([labels, new_labels], dim=0)
        else:
            final_params = params
            final_labels = labels

        return final_params, final_labels

    def train(self) -> None:
        num_epochs = self.config.get("num_epochs", 100)
        start_epoch = self.epoch
        use_interpolation = self.config.get("use_interpolation", True)

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            self.net.train()
            epoch_losses = []

            for batch in self.train_loader:
                embeddings, labels = batch
                embeddings = embeddings.cuda()
                labels = labels.cuda()

                if use_interpolation:
                    embeddings, labels = self.interpolate(embeddings, labels)

                pred = self.net(embeddings)
                loss = F.cross_entropy(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_losses.append(loss.item())

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})

                self.global_step += 1

            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.4f}")

            # Validation
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                train_acc = self.validate("train")
                val_acc = self.validate("val")
                print(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                self.save_ckpt()

            # Final test
            if epoch == num_epochs - 1:
                predictions, true_labels = self.validate("test", best=True, return_predictions=True)
                self.log_confusion_matrix(predictions, true_labels)

    @torch.no_grad()
    def validate(
        self,
        split: str,
        best: bool = False,
        return_predictions: bool = False
    ) -> Any:
        acc = AccuracyMetric()
        predictions = []
        true_labels = []

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        if best and self.best_model is not None:
            model = self.best_model
        else:
            model = self.net
        model = model.cuda()
        model.eval()

        losses = []
        for batch in loader:
            embeddings, labels = batch
            embeddings = embeddings.cuda()
            labels = labels.cuda()

            pred = model(embeddings)
            loss = F.cross_entropy(pred, labels)
            losses.append(loss.item())

            pred_softmax = F.softmax(pred, dim=-1)
            acc(pred_softmax, labels)

            predictions.append(pred_softmax.clone())
            true_labels.append(labels.clone())

        accuracy = acc.compute()

        self.logfn({f"{split}/acc": accuracy})
        self.logfn({f"{split}/loss": torch.mean(torch.tensor(losses))})

        if accuracy > self.best_acc and split == "val":
            self.best_acc = accuracy
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.net)

        if return_predictions:
            return torch.cat(predictions, dim=0), torch.cat(true_labels, dim=0)
        return accuracy.item()

    def log_confusion_matrix(self, predictions: Tensor, labels: Tensor) -> None:
        if wandb.run is not None:
            conf_matrix = wandb.plot.confusion_matrix(
                probs=predictions.cpu().numpy(),
                y_true=labels.cpu().numpy(),
                class_names=[str(i) for i in range(self.num_classes)],
            )
            self.logfn({"conf_matrix": conf_matrix})

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_acc": self.best_acc,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }

        torch.save(ckpt, self.ckpts_path / "last.pt")

        if best:
            torch.save(ckpt, self.ckpts_path / "best.pt")

    def restore_from_last_ckpt(self) -> None:
        ckpt_path = self.ckpts_path / "last.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = ckpt.get("global_step", self.epoch * len(self.train_loader))
            self.best_acc = ckpt["best_acc"]

            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])

            print(f"Restored from epoch {self.epoch - 1}, best acc: {self.best_acc:.4f}")


def main(config: Dict[str, Any] = None) -> None:
    if config is None:
        config = {
            "embeddings_path": "outputs/image_inr2vec/embeddings/embeddings.pt",
            "split_points": [55000, 60000],
            "layers_dim": [256, 256, 128],
            "num_classes": 10,
            "train_bs": 128,
            "val_bs": 128,
            "lr": 1e-3,
            "wd": 1e-4,
            "num_epochs": 100,
            "use_interpolation": True,
            "out_dir": "outputs/image_classifier",
        }

    # Initialize wandb if not already initialized
    if wandb.run is None:
        wandb.init(
            project="inr2vec-image-classifier",
            config=config,
        )

    trainer = EmbeddingClassifier(config)
    trainer.train()


if __name__ == "__main__":
    main()
