import sys

sys.path.append("..")

from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import pathlib
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# from pycarus.learning.models.siren import SIREN
from basemodels.inr2vec.models.encoder import Encoder
from basemodels.inr2vec.pcd_retrieval_and_classification.siren_utils import SIREN
from basemodels.inr2vec.utils import get_mlp_params_as_matrix
from basemodels.training import unwrap_nested_dictconfig


class InrDataset(Dataset):
    def __init__(self, inrs_root: Path, split: str, sample_sd: Dict[str, Any]) -> None:
        super().__init__()

        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_sd = sample_sd

    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            params = np.array(f.get("params"))
            params = torch.from_numpy(params).float()
            matrix = get_mlp_params_as_matrix(params, self.sample_sd)
            class_id = torch.from_numpy(np.array(f.get("class_id"))).long()

        return pcd, matrix, class_id


# @hmain(
#     base_cfg_dir="cfg/bases",
#     template_cfg_file="cfg/export_embeddings.yaml",
#     create_out_dir=False,
# )
# def unwrap_nested_dictconfig(cfg):
#     if isinstance(cfg, DictConfig):
#         # _data 안에 _data가 있으면 unwrap
#         if '_data' in cfg._data:
#             return unwrap_nested_dictconfig(cfg._data['_data'])
#         # 정상적인 DictConfig면 내부 값들도 재귀적으로 처리
#         result = {}
#         for key, value in cfg._data.items():
#             result[key] = unwrap_nested_dictconfig(value)
#         return DictConfig(result)
#     elif isinstance(cfg, list):
#         return [unwrap_nested_dictconfig(item) for item in cfg]
#     else:
#         return cfg

def export_embeddings(config) -> None:
    ckpt_path = config.get('ckpt_path')
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    saved_config = ckpt.get("config", config)
    saved_config = unwrap_nested_dictconfig(saved_config)

    siren_path = saved_config.get("siren_path")
    mlp_hdim = saved_config["mlp"]["hidden_dim"]
    num_hidden_layers = saved_config["mlp"]["num_hidden_layers"]

    mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
    sample_sd = mlp.state_dict()

    train_dset = InrDataset(siren_path, "train", sample_sd)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = InrDataset(siren_path, "val", sample_sd)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = InrDataset(siren_path, "test", sample_sd)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    encoder_cfg = saved_config.get("encoder", Dict[str, Any])
    encoder = Encoder(
        mlp_hdim,
        encoder_cfg["hidden_dims"],
        encoder_cfg["embedding_dim"],
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    loaders = [train_loader, val_loader, test_loader]
    splits = ["train", "val", "test"]

    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in loader:
            pcds, matrices, class_ids = batch
            matrices = matrices.cuda()

            with torch.no_grad():
                embedding = encoder(matrices)

            h5_path = Path(config["out_path"]) / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("pcd", data=pcds[0].detach().cpu().numpy())
                f.create_dataset("embedding", data=embedding[0].detach().cpu().numpy())
                f.create_dataset("class_id", data=class_ids[0].detach().cpu().numpy())

            idx += 1


# if __name__ == "__main__":
#     main()
