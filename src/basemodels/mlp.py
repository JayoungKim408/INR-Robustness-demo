import os
import torch
from typing import Dict, Union, Optional
from utils.config_utils import DictConfig
from utils.wandb_manager import WandbManager
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from basemodels.utils.nfn.data_utils import get_SirenDataset
from torch.utils.data import DataLoader, ConcatDataset
from attacks.surrogate.utils import get_inr_param_count_from_shapes


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1000, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.flatten_wsfeat(x)
        return self.net(x)

    def flatten_wsfeat(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        batch_size = wsfeat.weights[0].shape[0]
        parts = []
        for w in wsfeat.weights:
            parts.append(w.reshape(batch_size, -1))
        for b in wsfeat.biases:
            parts.append(b.reshape(batch_size, -1))
        return torch.cat(parts, dim=1)


@torch.no_grad()
def evaluate(classifier, loader, device):
    classifier.eval()
    correct, total = 0, 0
    for wts_and_bs, _, labels in loader:
        wsfeat = WeightSpaceFeatures(*wts_and_bs).to(device)
        labels = labels.to(device)
        preds = classifier(wsfeat).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main(cfg: Union[Dict, DictConfig],
         wandb_manager: Optional[WandbManager] = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    batch_size = cfg.get('batch_size', 32)
    loaders = {}
    dset_cfg = cfg.get('dset', {})
    trainset = get_SirenDataset(**dset_cfg, split='train')
    extra_aug = cfg.get('extra_aug', False)
    n_extra_aug = cfg.get('n_extra_aug', 10)
    if extra_aug:
        aug_dsets = []
        for i in range(1, n_extra_aug):

            aug_dsets.append(get_SirenDataset(
                siren_path=dset_cfg.get('siren_path'),
                siren_prefix=f"{dset_cfg.get('siren_prefix')}_{i}",
                split_points=dset_cfg.get('split_points'),
                split="train",
            ))

        trainset = ConcatDataset([trainset] + aug_dsets)
    loaders["train"] = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    valset = get_SirenDataset(**dset_cfg, split='val')
    loaders["val"] = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    testset = get_SirenDataset(**dset_cfg, split='test')
    loaders["test"] = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    print(f"Dataset sizes: train={len(trainset)}, val={len(valset)}, test={len(testset)}, batch_size={batch_size}.")

    # Infer input dim from data
    sample_wsfeat = WeightSpaceFeatures(*next(iter(loaders["train"]))[0])
    weight_shapes = [tuple(w.shape[1:]) for w in sample_wsfeat.weights]
    bias_shapes = [tuple(b.shape[1:]) for b in sample_wsfeat.biases]
    total_params = get_inr_param_count_from_shapes(weight_shapes, bias_shapes)
    print(f"INR params per sample: {total_params}")

    # Model
    classifier = MLPClassifier(
        input_dim=total_params,
        num_classes=cfg.get('num_classes', 10),
        hidden_dim=cfg.get('hidden_dim', 1000),
        num_layers=cfg.get('num_layers', 3),
    ).to(device)

    opt = torch.optim.Adam(classifier.parameters(), lr=cfg.get('lr', 1e-4))
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    best_acc = 0.0
    save_path = os.path.join(cfg.save_dir, "best.pt")

    for epoch in range(cfg.epochs):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        for wts_and_bs, _, labels in loaders["train"]:
            wsfeat = WeightSpaceFeatures(*wts_and_bs).to(device)
            labels = labels.to(device)

            preds = classifier(wsfeat)
            loss = criterion(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        train_acc = correct / total
        val_acc = evaluate(classifier, loaders["val"], device)
        print(f"[{epoch+1}/{cfg.epochs}] loss={total_loss/len(loaders['train']):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), save_path)
            print(f"  -> saved (best val_acc={best_acc:.4f})")

    # Test
    classifier.load_state_dict(torch.load(save_path, weights_only=False))
    test_acc = evaluate(classifier, loaders["test"], device)
    print(f"Test Accuracy: {test_acc:.4f}")