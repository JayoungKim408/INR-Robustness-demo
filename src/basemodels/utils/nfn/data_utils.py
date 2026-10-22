import typing
import glob
import re
import random
import os
import h5py
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Dataset, ConcatDataset, Subset, Sampler
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import pandas as pd
import PIL.Image as Image
from .nfn.common import state_dict_to_tensors
from inr.make_inr_init import flat_to_params_dict


def cycle(loader):
    while True:
        for blah in loader:
            yield blah


class StatTracker:
    def __init__(self):
        self.sum, self.sq_sum, self.count = 0, 0, 0

    def update(self, x: torch.Tensor):
        # x is a tensor of shape (batch_size, ...)
        self.sum += x.sum(dim=0)
        self.sq_sum += (x ** 2).sum(dim=0)
        self.count += x.shape[0]

    def get_mean_std(self):
        mean = self.sum / self.count
        std = torch.sqrt(self.sq_sum / self.count - mean ** 2)
        return mean, std

    
def compute_mean_std(loader, max_batches=None):
    w_trackers, b_trackers = None, None
    i = 0
    for params, *_ in loader:
        if max_batches is not None and i >= max_batches:
            break
        weights, bias = params
        if w_trackers is None:
            w_trackers = [StatTracker() for _ in weights]
            b_trackers = [StatTracker() for _ in bias]
        for w, w_tracker in zip(weights, w_trackers): w_tracker.update(w)
        for b, b_tracker in zip(bias, b_trackers): b_tracker.update(b)
        i += 1
    w_stats = [w_tracker.get_mean_std() for w_tracker in w_trackers]
    b_stats = [b_tracker.get_mean_std() for b_tracker in b_trackers]
    return w_stats, b_stats


class ZooDataset(Dataset):
    def __init__(self, data_path, mode, idcs_file=None):
        data = np.load(os.path.join(data_path, "weights.npy"))
        # Hardcoded shuffle order for consistent test set.
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        metrics = pd.read_csv(os.path.join(data_path, "metrics.csv.gz"), compression='gzip')
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(os.path.join(data_path, "layout.csv"))
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[mode]
        data = data[idcs]
        self.metrics = metrics.iloc[idcs]

        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def __getitem__(self, idx):
        # insert a channel dim
        weights = tuple(w[idx][None] for w in self.weights)
        biases = tuple(b[idx][None] for b in self.biases)
        return (weights, biases), self.metrics.iloc[idx].test_accuracy.item()
    
def get_SirenDataset(siren_path, siren_prefix="init_pool", split="all", split_points=None, **kwargs):
    if "shapenet" in str(siren_path):
        return ShapeNetSirenDataset(siren_path=siren_path, siren_prefix=siren_prefix, split=split)
    return SirenDataset(siren_path=siren_path, siren_prefix=siren_prefix, split=split, split_points=split_points, **kwargs)

# class ShapeNetSirenDataset(Dataset):
#     def __init__(self, 
#                  siren_path, 
#                  split='train', 
#     ): 
#         self.inrs_root = Path(siren_path) / split
#         self.h5_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))

#         # Init weights (shared across all samples)
#         # init_path = Path(siren_path) / "inits" / "in3_out1_h512_l4.pt"
#         init_path = Path(siren_path) / "inits" / "in3_out1_h128_l1.pt"
#         init_sd = torch.load(init_path, map_location="cpu")
#         self.init_weights, self.init_biases = state_dict_to_tensors(init_sd)

#         # param_shapes from init state dict
#         self.param_shapes = [(k, v.shape) for k, v in init_sd.items()]

#     def __len__(self):
#         return len(self.h5_paths)

#     def __getitem__(self, idx):
#         with h5py.File(self.h5_paths[idx], "r") as f:
#             params_flat = torch.from_numpy(np.array(f["params"])).float()
#             label = int(f["class_id"][()])

#         params_dict = flat_to_params_dict(params_flat, self.param_shapes)
#         weights, biases = state_dict_to_tensors(params_dict)

#         return (weights, biases), (self.init_weights, self.init_biases), label
    
class ShapeNetSirenDataset(Dataset):
    def __init__(self, siren_path, siren_prefix="init_pool", split='train'):
        self.inrs_root = Path(siren_path) / siren_prefix / split
        self.h5_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))

        init_dir = Path(siren_path) / siren_prefix / "inits"
        self.shared_init = not siren_prefix.startswith("diff_init")
        # self.num_query_points = num_query_points

        if self.shared_init:
            init_path = next((p for p in [init_dir / "in3_out1_h512_l4.pt"] if p.exists()), None)
            init_sd = torch.load(init_path, map_location="cpu")
            self.init_weights, self.init_biases = state_dict_to_tensors(init_sd)
            self.param_shapes = [(k, v.shape) for k, v in init_sd.items()]
        else:
            # Get param_shapes from model definition
            from basemodels.inr2vec.pcd_retrieval_and_classification.siren_utils import SIREN
            # from pycarus.learning.models.siren import SIREN
            model = SIREN(input_dim=3, hidden_dim=128, num_hidden_layers=2, out_dim=1)
            sd = model.state_dict()
            self.param_shapes = [(k, v.shape) for k, v in sd.items()]

    def __len__(self):
        return len(self.h5_paths)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_paths[idx], "r") as f:
            params_flat = torch.from_numpy(np.array(f["params"])).float()
            label = int(f["class_id"][()])

            if not self.shared_init:
                init_flat = torch.from_numpy(np.array(f["init_params"])).float()

        params_dict = flat_to_params_dict(params_flat, self.param_shapes)
        weights, biases = state_dict_to_tensors(params_dict)

        if self.shared_init:
            init_weights, init_biases = self.init_weights, self.init_biases
        else:
            init_dict = flat_to_params_dict(init_flat, self.param_shapes)
            init_weights, init_biases = state_dict_to_tensors(init_dict)

        return (weights, biases), (init_weights, init_biases), label

DEF_TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
class SirenDataset(Dataset):
    def __init__(
        self,
        siren_path,
        siren_prefix="init_pool",
        split="all",
        split_points: typing.Tuple[int, int] = None,
    ):
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"/(\d)s/"
        self.idx_to_path = {}
        self.idx_to_label = {}
        self.param_shape = torch.load(os.path.join(siren_path, "param_shapes.pth"), map_location="cpu", weights_only=False)

        for p in glob.glob(os.path.join(siren_path, f"{siren_prefix}/[0-9]s/*.pth")):
            idx = int(re.search(idx_pattern, p).group(1))
            self.idx_to_path[idx] = p
            label = int(re.search(label_pattern, p).group(1))
            self.idx_to_label[idx] = label

        if split == "all":
            self.idcs = list(range(len(self.idx_to_path)))
        else:
            val_point, test_point = split_points
            all_idcs = sorted(self.idx_to_path.keys())
            self.idcs = {
                "train": [i for i in all_idcs if i < val_point],
                "val": [i for i in all_idcs if val_point <= i < test_point],
                "test": [i for i in all_idcs if i >= test_point],
            }[split]

    def __getitem__(self, idx):
        data_idx = self.idcs[idx]

        with open(self.idx_to_path[data_idx], "rb") as f:
            sd = torch.load(f, weights_only=False, map_location="cpu")
        if 'label' in sd.keys():
            del sd['label']

        params, init = sd['params'], sd['init']
        params = flat_to_params_dict(params, self.param_shape)
        init = flat_to_params_dict(init, self.param_shape)
        weights, biases = state_dict_to_tensors(params)
        init_weights, init_biases = state_dict_to_tensors(init)
        label = self.idx_to_label[data_idx]

        return (weights, biases), (init_weights, init_biases), label

    def __len__(self):
        return len(self.idcs)

def _load_image_dataset(siren_path, data_path, data_tfm=DEF_TFM):
    if "mnist" in siren_path:
        data_type = "mnist"
        train = MNIST(data_path, transform=data_tfm, train=True, download=True)
        test = MNIST(data_path, transform=data_tfm, train=False, download=True)
    elif "fashion" in siren_path:
        data_type = "fashion"
        train = FashionMNIST(data_path, transform=data_tfm, train=True, download=True)
        test = FashionMNIST(data_path, transform=data_tfm, train=False, download=True)
    else:
        data_type = "cifar"
        train = CIFAR10(data_path, transform=data_tfm, train=True, download=True)
        test = CIFAR10(data_path, transform=data_tfm, train=False, download=True)
    return ConcatDataset([train, test]), data_type


def _split_indices(n, split, split_points):
    if split == "all":
        return list(range(n))
    val_point, test_point = split_points
    return {
        "train": list(range(val_point)),
        "val": list(range(val_point, test_point)),
        "test": list(range(test_point, n)),
    }[split]

def get_SirenAdnOriginalDataset(siren_path, siren_prefix="init_pool", data_path=None, split="all", split_points=None, **kwargs):
    if "shapenet" in str(siren_path):
        return ShapeNetSirenAndOriginalDataset(siren_path=siren_path, siren_prefix=siren_prefix, split=split)
    return SirenAndOriginalDataset(siren_path=siren_path, siren_prefix=siren_prefix, data_path=data_path, split=split, split_points=split_points, **kwargs)

class ShapeNetSirenAndOriginalDataset(ShapeNetSirenDataset):
    def __init__(self, siren_path, siren_prefix="init_pool", split="train"):
        super().__init__(siren_path=siren_path, siren_prefix=siren_prefix, split=split)

    def __getitem__(self, idx):
        params, init_params, label = super().__getitem__(idx)

        with h5py.File(self.h5_paths[idx], "r") as f:
            pcd = torch.from_numpy(np.array(f["pcd"])).float()

        return params, init_params, pcd, label

class SirenAndOriginalDataset(Dataset):
    def __init__(
        self,
        siren_path,
        siren_prefix,
        data_path,
        split="all",
        data_tfm=DEF_TFM,
        split_points=None,
    ):
        siren_dset = SirenDataset(siren_path, split="all", siren_prefix=siren_prefix)
        dset, self.data_type = _load_image_dataset(siren_path, data_path, data_tfm)

        self.idcs = _split_indices(len(siren_dset), split, split_points)
        self.siren_dset = Subset(siren_dset, self.idcs)
        self.dset = Subset(dset, self.idcs)

    def __len__(self):
        return len(self.siren_dset)

    def __getitem__(self, idx):
        params, init_params, siren_label = self.siren_dset[idx]
        img, data_label = self.dset[idx]
        assert siren_label == data_label
        return params, init_params, img, siren_label

def get_SirenOriginalAndEmbeddingDataset(siren_path, siren_prefix="init_pool", data_path=None, embedding_path=None, split="all", split_points=None, **kwargs):
    if "shapenet" in str(siren_path):
        return ShapeNetSirenOriginalAndEmbeddingDataset(siren_path=siren_path, siren_prefix=siren_prefix, embedding_path=embedding_path, split=split)
    return SirenOriginalAndEmbeddingDataset(siren_path=siren_path, siren_prefix=siren_prefix, data_path=data_path, embedding_path=embedding_path, split=split, split_points=split_points, **kwargs)

class SirenOriginalAndEmbeddingDataset(SirenAndOriginalDataset):
    def __init__(self, siren_path, siren_prefix, data_path,
                 embedding_path, split="all", data_tfm=DEF_TFM,
                 split_points=None):
        super().__init__(siren_path, siren_prefix, data_path,
                         split=split, data_tfm=data_tfm,
                         split_points=split_points)
        embeddings = torch.load(embedding_path, weights_only=False)["embeddings"]
        self.embeddings = embeddings[self.idcs]

    def __getitem__(self, idx):
        params, init_params, img, label = super().__getitem__(idx)
        embedding = self.embeddings[idx]
        return params, init_params, img, embedding, label

class ShapeNetSirenOriginalAndEmbeddingDataset(ShapeNetSirenAndOriginalDataset):
    def __init__(self, siren_path, siren_prefix, embedding_path, split="train"):
        super().__init__(siren_path=siren_path, siren_prefix=siren_prefix, split=split)
        self.emb_root = Path(embedding_path) / split
        self.emb_paths = sorted(self.emb_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_wise_embs = len(self.emb_paths) > 0
        
        if self.sample_wise_embs: 
            assert len(self.emb_paths) == len(self.h5_paths), \
                f"Embedding count ({len(self.emb_paths)}) != INR count ({len(self.h5_paths)})"
        else:
            ckpt = torch.load(Path(embedding_path)/"embeddings.pt", weights_only=False)
            embeddings, split_points = ckpt["embeddings"], ckpt["split_points"]
            self.idcs = _split_indices(len(embeddings), split, split_points)
            self.embeddings = embeddings[self.idcs]

    def __getitem__(self, idx):
        params, init_params, pcd, label = super().__getitem__(idx)
        if self.sample_wise_embs:
            with h5py.File(self.emb_paths[idx], "r") as f:
                embedding = torch.from_numpy(np.array(f["embedding"])).float()
        else:
            embedding = self.embeddings[idx]
        return params, init_params, pcd, embedding, label


class AlignedSampler(Sampler):
    def __init__(self, data_source, samples_per_dset):
        """Assumes data_source is a concatenation of N datasets, each of size samples_per_dset."""
        self.data_source = data_source
        assert len(self.data_source) % samples_per_dset == 0, "Dataset size must be divisible by samples_per_dset."
        self.num_dsets = len(self.data_source) // samples_per_dset
        self.samples_per_dset = samples_per_dset

    def __iter__(self):
        example_indices = torch.randperm(self.samples_per_dset).tolist()
        # get the example_indices[i]th example from each dataset, for i in range(samples_per_dset)
        full_indices = []
        for idx in example_indices:
            full_indices.extend([idx + i * self.samples_per_dset for i in range(self.num_dsets)])
        yield from full_indices

    def __len__(self):
        return len(self.data_source)



def combine_images_side_by_side(image1, image2, image3, if_norm=True):
    """
    Combine three PIL Image objects into one, side by side.

    Parameters:
    - image1: PIL Image object.
    - image2: PIL Image object.
    - image3: PIL Image object.

    Returns:
    - A new PIL Image object containing the three images side by side.
    """
    if if_norm:
        image1 = (image1 + 1) / 2
        image2 = (image2 + 1) / 2
        image3 = (image3 + 1) / 2
    image1 = (image1 * 255).to(torch.uint8)
    image2 = (image2 * 255).to(torch.uint8)
    image3 = (image3 * 255).to(torch.uint8)

    # Calculate the total width and the maximum height
    total_width = image1.width + image2.width + image3.width
    max_height = max(image1.height, image2.height, image3.height)

    # Create a new image with the calculated dimensions
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste the images into the new image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    combined_image.paste(image3, (image1.width + image2.width, 0))

    return combined_image
