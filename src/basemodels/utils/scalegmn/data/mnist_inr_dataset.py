import os
import glob
import re
from typing import Tuple
from .base_datasets import BaseDataset
from torchvision import transforms
import numpy as np
import torchvision
from pathlib import Path

from ..utils.image_processing import style_edit

class LabeledINRDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            dataset_path,
            split_path,
            debug=False,
            split="train",
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            image_size=(28,28),
            layer_layout=None,
            direction='forward',
            return_path=False,
            data_format="graph",
            switch_to_canon=True
    ):
        super().__init__(
            dataset,
            dataset_path,
            split_path,
            split,
            node_pos_embed,
            edge_pos_embed,
            equiv_on_hidden,
            get_first_layer_mask,
            image_size,
            layer_layout,
            direction,
            return_path,
            data_format,
            switch_to_canon)

        if debug:
            self.dataset = {k: v[:16] for k,v in self.dataset.items()}

    def __len__(self):
        return len(self.dataset["label"])

    def get_path(self, index):
        return self.dataset["path"][index], None
    
    def get_label(self, index, state_dict, aux):
        return int(self.dataset["label"][index])


class MNISTINRImageDataset(LabeledINRDataset):
    def __init__(self, img_ds, style_function, **kwargs):
        super().__init__(**kwargs)
        self.dataset["path"] = [Path(kwargs['dataset_path'] + '/'.join(p.split("/")[2:])) for p in self.dataset["path"]]
        self.img_ds = getattr(torchvision.datasets, 'MNIST')(
            root=kwargs['dataset_path'],
            train=(self.split in ['train', 'val']),
            download=True,
            transform=None)

        transformation = style_edit(style_function)
        self.img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                transformation,
                transforms.ToTensor(),
                # transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ]
        )
        self.input_img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, item):
        batch, w_b = super().__getitem__(item)
        img_id = int(self.dataset["path"][item].parts[-3].split("_")[-1])
        img, _ = self.img_ds[img_id]
        transformed_img = self.img_transform(img)
        input_img = self.input_img_transform(img)
        return batch, w_b, transformed_img, input_img



class MNISTINRDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            dataset_path,
            split_path="",
            debug=False,
            split="train",
            split_points: Tuple[int, int] = None,
            prefix="randinit_smaller",
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            image_size=(28, 28),
            direction='forward',
            layer_layout=None,
            return_path=False,
            data_format="graph",
            switch_to_canon=True
    ):
        self.idx_to_path = {}
        self.idx_to_label = {}
        self.prefix = prefix
        self.split_points = split_points

        super().__init__(
            dataset,
            dataset_path,
            split_path,
            split,
            node_pos_embed,
            edge_pos_embed,
            equiv_on_hidden,
            get_first_layer_mask,
            image_size,
            layer_layout,
            direction,
            return_path,
            data_format,
            switch_to_canon)

        if debug:
            self.dataset = self.dataset[:16]

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, split_path):
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"/(\d)s/"

        for siren_path in glob.glob(os.path.join(self.dataset_path, f"{self.prefix}/[0-9]s/*.pth")):
            idx = int(re.search(idx_pattern, siren_path).group(1))
            self.idx_to_path[idx] = siren_path
            label = int(re.search(label_pattern, siren_path).group(1))
            self.idx_to_label[idx] = label
        if self.split == "all":
            dataset = list(range(len(self.idx_to_path)))
        else:
            val_point, test_point = self.split_points
            dataset = {
                "train": list(range(val_point)),
                "val": list(range(val_point, test_point)),
                "test": list(range(test_point, len(self.idx_to_path))),
            }[self.split]
        return dataset

    def get_path(self, index):
        data_idx = self.dataset[index]
        path = self.idx_to_path[data_idx]
        return path, data_idx

    def get_label(self, index, state_dict, data_idx):
        label = self.idx_to_label[data_idx]
        return label