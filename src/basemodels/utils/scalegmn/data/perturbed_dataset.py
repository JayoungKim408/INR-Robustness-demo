from .base_datasets import BaseDataset
from torch.utils.data import ConcatDataset
import torch
import os
import glob
import re
from typing import Tuple
from torchvision import transforms
from torchvision.datasets import CIFAR10,  FashionMNIST, MNIST

class INRPerturbedDataset(BaseDataset):
    """
    Dataset that returns both clean INR graph and perturbed counterparts (x_adv image and INR-adv graph),
    similar in spirit to NFN's PerturbedDataset, while keeping the original CifarINRDataset untouched.
    """
    def __init__(
            self,
            type,
            dataset,
            dataset_path,
            split_path="",
            debug=False,
            split="test",
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
            switch_to_canon=True,
            *,
            x_adv_path: str,
            inr_adv_path: str,
            inr_adv_prefix: str = "inr_adv",
            raw_data_path: str = None,
    ):
        assert split == "test", "INRPerturbedDataset is intended to be used with split='test'."
        self.idx_to_path = {}
        self.idx_to_label = {}
        self.prefix = prefix
        self.split_points = [45000, 50000] if dataset == "cifar" else [55000, 60000]
        self._inr_adv_path = inr_adv_path
        self._inr_adv_prefix = inr_adv_prefix
        self._raw_data_path = raw_data_path or "/nfs/stak/users/hongsa/INR-Robustness/raw"

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

        DEF_TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        DATA_MAP = {'cifar': CIFAR10, 'fashion': FashionMNIST, 'mnist': MNIST}
        # Load x_advs tensor once
        self.x_advs = torch.load(x_adv_path, map_location="cpu", weights_only=False)
        # Build clean raw image dataset (train + test concat to align global indices with siren idx)
        cifar_train = DATA_MAP[dataset](self._raw_data_path, transform=DEF_TFM, train=True, download=True)
        cifar_test = DATA_MAP[dataset](self._raw_data_path, transform=DEF_TFM, train=False, download=True)
        self._raw_dset = ConcatDataset([cifar_train, cifar_test])
        # Build mapping from adv index -> inr_adv file path (expects suffix index)
        self.perturbed_idx_to_path = {}
        for p in glob.glob(os.path.join(inr_adv_path, f"{inr_adv_prefix}_*.pt")):
            fname = os.path.basename(p)
            stem, _ = os.path.splitext(fname)
            # capture last numeric suffix (works with eps embedded like inr_adv_0.06_123.pt)
            parts = stem.split("_")
            if parts and parts[-1].isdigit():
                adv_idx = int(parts[-1])
                self.perturbed_idx_to_path[adv_idx] = p

        if debug:
            self.dataset = self.dataset[:16]

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, split_path):
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"_(\d)s"

        for siren_path in glob.glob(os.path.join(self.dataset_path, f"{self.prefix}_[0-9]s/*.pth")):
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

    def __getitem__(self, index):
        """
        Returns:
            x_clean:           clean image tensor (C,H,W) normalized to [-1,1]
            inr_clean_graph:   PyG graph constructed from clean INR
            x_adv:             adversarial image tensor (C,H,W) in the saved scale
            inr_adv_graph:     PyG graph constructed from adversarial INR
            label:             ground-truth class label (int)
        """
        # -- Clean INR (reuse BaseDataset logic) --
        path, data_idx = self.get_path(index)
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        label = self.get_label(index, state_dict, data_idx)

        weights = tuple([v.permute(1, 0) for w, v in state_dict.items() if "weight" in w])
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        node_features, edge_features = self.batch_to_graphs(weights, biases)
        # Ensure node feature dimension matches models trained with 1-d node feature
        if hasattr(node_features, "shape") and node_features.shape[-1] > 1:
            node_features = node_features[:, :1]
        inr_clean_graph, _ = self._to_pyg(node_features, edge_features, label)

        # -- Adv INR and x_adv --
        # Map from global clean idx to adv index (offset by end of val set)
        _, test_point = self.split_points
        adv_idx = data_idx - test_point
        assert adv_idx >= 0, f"Computed adv_idx {adv_idx} < 0; check split_points and data_idx."
        assert adv_idx in self.perturbed_idx_to_path, f"Adv INR not found for adv_idx={adv_idx}"
        adv_sd_path = self.perturbed_idx_to_path[adv_idx]
        adv_sd = torch.load(adv_sd_path, map_location="cpu", weights_only=False)

        adv_weights = tuple([v.permute(1, 0) for w, v in adv_sd.items() if "weight" in w])
        adv_weights = tuple([w.unsqueeze(-1) for w in adv_weights])
        adv_biases = tuple([v for w, v in adv_sd.items() if "bias" in w])
        adv_biases = tuple([b.unsqueeze(-1) for b in adv_biases])

        adv_node_features, adv_edge_features = self.batch_to_graphs(adv_weights, adv_biases)
        if hasattr(adv_node_features, "shape") and adv_node_features.shape[-1] > 1:
            adv_node_features = adv_node_features[:, :1]
        inr_adv_graph, _ = self._to_pyg(adv_node_features, adv_edge_features, label)

        # x_adv tensor aligns with adv_idx order saved during generation
        assert adv_idx < len(self.x_advs), f"x_advs out of range for adv_idx={adv_idx}"
        x_adv = self.x_advs[adv_idx]
        # Ensure collate can stack without autograd complaining
        if isinstance(x_adv, torch.Tensor):
            x_adv = x_adv.detach().clone().requires_grad_(False)
        else:
            x_adv = torch.as_tensor(x_adv).detach().clone().requires_grad_(False)

        # Clean image aligned to global index
        x_clean, _ = self._raw_dset[data_idx]
        if isinstance(x_clean, torch.Tensor):
            x_clean = x_clean.detach().clone().requires_grad_(False)
        else:
            x_clean = torch.as_tensor(x_clean).detach().clone().requires_grad_(False)

        return x_clean, inr_clean_graph, x_adv, inr_adv_graph, label

    def _to_pyg(self, node_features, edge_features, label):
        from .data_utils import to_pyg_batch, get_node_types, get_edge_types, nn_to_edge_index
        batch, _ = to_pyg_batch(
            node_features,
            edge_features,
            self.edge_index,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            direction=self.direction,
            label=label,
            hidden_nodes=self.hidden_nodes if self.equiv_on_hidden else None,
            first_layer_nodes=self.first_layer_nodes if self.get_first_layer_mask else None
        )
        return batch, label