import sys
sys.path.append("..")
import h5py
import torch
from torch import Tensor
from pathlib import Path
from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader, ConcatDataset

from basemodels.inr2vec.models.encoder import Encoder
from basemodels.utils.nfn.data_utils import get_SirenAdnOriginalDataset


def get_mlp_params_as_matrix_from_wsfeat(weights: Tuple, biases: Tuple, hidden_dim: int) -> Tensor:
    """Convert weights/biases tuple format to matrix format for encoder."""
    num_layers = len(weights)

    if num_layers <= 2:
        params = []
        for w, b in zip(weights, biases):
            params.append(w.squeeze(0).reshape(-1))
            params.append(b.squeeze(0).reshape(-1))
        params = torch.cat(params)
        return params.reshape(-1, hidden_dim)

    params = []
    for i in range(1, num_layers - 1):
        w = weights[i].squeeze(0)
        b = biases[i].squeeze(0)
        layer_params = torch.cat([w, b.unsqueeze(-1)], dim=-1)
        params.append(layer_params)

    if len(params) == 0:
        params = []
        for w, b in zip(weights, biases):
            params.append(w.squeeze(0).reshape(-1))
            params.append(b.squeeze(0).reshape(-1))
        params = torch.cat(params)
        return params.reshape(-1, hidden_dim)

    params = torch.cat(params, dim=0)
    return params


class ImageInrDatasetForExport:
    """Minimal dataset for exporting embeddings."""

    def __init__(
        self,
        siren_path: str,
        siren_prefix: str,
        data_path: str,
        split: str = "all",
        split_points: Tuple[int, int] = None,
        hidden_dim: int = 32,
    ) -> None:
        self.base_dataset = get_SirenAdnOriginalDataset(
            siren_path=siren_path,
            siren_prefix=siren_prefix,
            data_path=data_path,
            split=split,
            split_points=split_points,
        )
        self.hidden_dim = hidden_dim
        self.data_type = self.base_dataset.data_type

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        params, init_params, img, label = self.base_dataset[index]
        weights, biases = params

        matrix = get_mlp_params_as_matrix_from_wsfeat(weights, biases, self.hidden_dim)

        return matrix, init_params, label


def export_embeddings(config: Dict[str, Any]) -> None:
    """Export embeddings from a trained encoder."""

    # Load checkpoint
    ckpt_path = Path(config["ckpt_path"])
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    # Get saved config
    saved_config = ckpt.get("config", config)

    # Dataset configuration
    siren_path = saved_config.get("siren_path")
    siren_prefix = saved_config.get("siren_prefix", "randinit_test")
    extra_aug = saved_config.get("extra_aug", False)
    n_extra_aug = saved_config.get("n_extra_aug", 0)
    data_path = saved_config.get("data_path")
    split_points = tuple(saved_config.get("split_points", [55000, 60000]))
    mlp_hdim = saved_config.get("mlp_hidden_dim", 32)
    inr_type = saved_config.get("inr_type", "siren")

    # Create dataset
    train_dset = ImageInrDatasetForExport(
        siren_path, siren_prefix, data_path,
        split="train", split_points=split_points, hidden_dim=mlp_hdim,
    )
    val_dset = ImageInrDatasetForExport(
        siren_path, siren_prefix, data_path,
        split="val", split_points=split_points, hidden_dim=mlp_hdim,
    )
    test_dset = ImageInrDatasetForExport(
        siren_path, siren_prefix, data_path,
        split="test", split_points=split_points, hidden_dim=mlp_hdim,
    )
    aug_dset = []
    if extra_aug:
        for i in range(1, n_extra_aug):
            aug_dset.append(ImageInrDatasetForExport(
                siren_path, f"{siren_prefix}_{i}", data_path,
                split="train", split_points=split_points, hidden_dim=mlp_hdim,
                inr_type=inr_type,
            ))
        split_points = [split_points[0] * n_extra_aug, split_points[0] * n_extra_aug + 5000]
    dset = ConcatDataset([train_dset] + aug_dset + [val_dset] + [test_dset])
    print(f"Dataset size: {len(dset)}, split_points: {split_points}")
    
    batch_size = saved_config.get("batch_size", 64)
    loader = DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Get matrix dimension from first sample
    sample_matrix, _, _ = dset[0]
    matrix_feat_dim = sample_matrix.shape[-1]

    # Build encoder
    encoder_cfg = saved_config.get("encoder", {})
    encoder = Encoder(
        matrix_feat_dim,
        encoder_cfg.get("hidden_dims", [128, 256, 512]),
        encoder_cfg.get("embedding_dim", 256),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    # Export embeddings
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            matrices, init_params, labels = batch
            matrices = matrices.cuda().float()

            embeddings = encoder(matrices)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save embeddings
    out_path = Path(config.get("out_path", "embeddings")) / "embeddings.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "embeddings": all_embeddings,
        "labels": all_labels,
        "split_points": split_points,
        "data_type": train_dset.data_type,
    }, out_path)

    print(f"Saved {len(all_embeddings)} embeddings to {out_path}")

    # Optionally save as h5 files (compatible with mesh_classification format)
    if config.get("save_h5", False):
        h5_dir = out_path.parent / "h5"

        for split_name, (start, end) in [
            ("train", (0, split_points[0])),
            ("val", (split_points[0], split_points[1])),
            ("test", (split_points[1], len(all_embeddings))),
        ]:
            split_dir = h5_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(start, end):
                with h5py.File(split_dir / f"{i - start}.h5", "w") as f:
                    f.create_dataset("embedding", data=all_embeddings[i].numpy())
                    f.create_dataset("class_id", data=all_labels[i].numpy())

        print(f"Saved h5 files to {h5_dir}")


def main(config: Dict[str, Any] = None) -> None:
    if config is None:
        config = {
            "ckpt_path": "outputs/image_inr2vec/ckpts/best.pt",
            "siren_path": "/path/to/siren/data",
            "siren_prefix": "randinit_test",
            "data_path": "/path/to/image/data",
            "split_points": [55000, 60000],
            "mlp_hidden_dim": 32,
            "batch_size": 64,
            "out_path": "outputs/image_inr2vec/embeddings/embeddings.pt",
            "save_h5": False,
        }

    export_embeddings(config)


if __name__ == "__main__":
    main()
