import logging
import os
from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from basemodels.utils.dws.experiments.data import INRDataset
from basemodels.utils.dws.experiments.utils import (
    common_parser,
    count_parameters,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from basemodels.utils.dws.nn.models import DWSModelForClassification, MLPModelForClassification

set_logger()


# Global variable to hold args when called from CLI
_cli_args = None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for batch in loader:
        batch = batch.to(device)
        inputs = (batch.weights, batch.biases)
        out = model(inputs)
        loss += F.cross_entropy(out, batch.label, reduction="sum")
        total += len(batch.label)
        pred = out.argmax(1)
        correct += pred.eq(batch.label).sum()
        predicted.extend(pred.cpu().numpy().tolist())
        gt.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(avg_loss=avg_loss, avg_acc=avg_acc, predicted=predicted, gt=gt)


def main(
    dataset_name,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    eval_every: int,
    base_path: str = "/nfs/stak/users/hongsa/INR-Robustness",
    save_dir: Optional[str] = None,
    siren_path: Optional[str] = None,  # Path to SIREN weights directory (for INR config support)
    # Training options
    normalize: bool = True,
    augmentation: bool = True,
    permutation: bool = False,
    n_samples: Optional[int] = None,
    num_workers: int = 8,
    # Model options
    model_type: str = "dwsnet",
    dim_hidden: int = 32,
    n_hidden: int = 4,
    reduction: str = "max",
    n_fc_layers: int = 1,
    set_layer: str = "sab",
    n_out_fc: int = 1,
    do_rate: float = 0.0,
    add_bn: bool = True,
    diagonal: bool = False,
    # Optimizer options
    optim: str = "adamw",
    # Logging
    use_wandb: bool = False,
):
    """
    Train DWS (Deep Weight Space) classifier.

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: PyTorch device
        eval_every: Evaluate every N epochs
        base_path: Base path for data and checkpoints
        save_dir: Directory to save checkpoints (defaults to base_path/basemodels/dws/{dataset_name})
        normalize: Whether to normalize data
        augmentation: Whether to use augmentation
        permutation: Whether to use permutation augmentation
        n_samples: Number of training samples (None = use all)
        num_workers: Number of data loader workers
        model_type: 'dwsnet' or 'mlp'
        dim_hidden: Hidden dimension
        n_hidden: Number of hidden layers
        reduction: Reduction strategy ('mean', 'sum', 'max')
        n_fc_layers: Number of FC layers per block
        set_layer: Set layer type ('sab', 'ds')
        n_out_fc: Number of output FC layers
        do_rate: Dropout rate
        add_bn: Whether to add batch normalization
        diagonal: Whether to use diagonal DWSNet
        optim: Optimizer ('adam', 'sgd', 'adamw')
        use_wandb: Whether to use wandb logging
    """
    # Use CLI args if available (for backward compatibility)
    global _cli_args
    if _cli_args is not None:
        normalize = _cli_args.normalize
        augmentation = _cli_args.augmentation
        permutation = _cli_args.permutation
        n_samples = _cli_args.n_samples
        num_workers = _cli_args.num_workers
        model_type = _cli_args.model
        dim_hidden = _cli_args.dim_hidden
        n_hidden = _cli_args.n_hidden
        reduction = _cli_args.reduction
        n_fc_layers = _cli_args.n_fc_layers
        set_layer = _cli_args.set_layer
        n_out_fc = _cli_args.n_out_fc
        do_rate = _cli_args.do_rate
        add_bn = _cli_args.add_bn
        diagonal = _cli_args.diagonal
        optim = _cli_args.optim
        use_wandb = _cli_args.wandb

    # Set default save_dir if not provided
    if save_dir is None:
        save_dir = os.path.join(base_path, "basemodels", "dws", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Build paths - use siren_path if provided, otherwise use default
    if siren_path is None:
        siren_path = os.path.join(base_path, f"sirens/siren_{dataset_name}_wts")
    path = os.path.join(siren_path, "splits.json")
    statistics_path = os.path.join(siren_path, "statistics.pth")

    # load dataset
    train_set = INRDataset(
        path=path,
        split="train",
        normalize=normalize,
        augmentation=augmentation,
        permutation=permutation,
        statistics_path=statistics_path,
    )
    val_set = INRDataset(
        path=path,
        split="val",
        normalize=normalize,
        statistics_path=statistics_path,
    )
    test_set = INRDataset(
        path=path,
        split="test",
        normalize=normalize,
        statistics_path=statistics_path,
    )

    if n_samples is not None:
        _, train_indices = train_test_split(
            range(len(train_set)), test_size=n_samples
        )
        train_set = torch.utils.data.Subset(train_set, train_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    logging.info(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}"
    )

    point = train_set.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    # todo: make defaults for MLP so that parameters for MLP and DWS are the same
    model = {
        "dwsnet": DWSModelForClassification(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=1,
            hidden_dim=dim_hidden,
            n_hidden=n_hidden,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            set_layer=set_layer,
            n_out_fc=n_out_fc,
            dropout_rate=do_rate,
            bn=add_bn,
            diagonal=diagonal,
        ).to(device),
        "mlp": MLPModelForClassification(
            in_dim=sum([w.numel() for w in weight_shapes + bias_shapes]),
            hidden_dim=dim_hidden,
            n_hidden=n_hidden,
            bn=add_bn,
        ).to(device),
    }[model_type]

    logging.info(f"number of parameters: {count_parameters(model)}")

    optimizer = {
        "adam": torch.optim.Adam(
            [
                dict(params=model.parameters(), lr=lr),
            ],
            lr=lr,
            weight_decay=5e-4,
        ),
        "sgd": torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9
        ),
        "adamw": torch.optim.AdamW(
            params=model.parameters(), lr=lr, amsgrad=True, weight_decay=5e-4
        ),
    }[optim]

    epoch_iter = trange(epochs)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch = batch.to(device)
            inputs = (batch.weights, batch.biases)
            out = model(inputs)

            loss = criterion(out, batch.label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            if use_wandb:
                log = {
                    "train/loss": loss.item(),
                }
                wandb.log(log)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
            )

        if (epoch + 1) % eval_every == 0:
            val_loss_dict = evaluate(model, val_loader, device)
            test_loss_dict = evaluate(model, test_loader, device)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            train_loss_dict = evaluate(model, train_loader, device)

            best_val_criteria = val_acc >= best_val_acc

            if best_val_criteria:
                best_val_acc = val_acc
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                ckpt_path = os.path.join(save_dir, "best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved model to {ckpt_path} at epoch {epoch}")

            if use_wandb:
                log = {
                    "train/avg_loss": train_loss_dict["avg_loss"],
                    "train/acc": train_loss_dict["avg_acc"],
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    # "test/conf_mat": wandb.plot.confusion_matrix(
                    #     probs=None,
                    #     y_true=test_loss_dict["gt"],
                    #     preds=test_loss_dict["predicted"],
                    #     class_names=range(10),
                    # ),
                    "epoch": epoch,
                }

                wandb.log(log)


if __name__ == "__main__":
    parser = ArgumentParser(
        "INR classification trainer", parents=[common_parser]
    )

    parser.set_defaults(
        # data_path="dataset/mnist_splits.json",
        n_epochs=100,
        batch_size=512,
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/nfs/stak/users/hongsa/INR-Robustness",
        help="base path for data and checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dwsnet",
        choices=["dwsnet", "mlp"],
        help="model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar",
        choices=["mnist", "cifar", "fashion"],
        help="dataset_name",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None, help="num training samples"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "adamw"],
        help="optimizer",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="num workers")
    parser.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["mean", "sum", "max"],
        help="reduction strategy",
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        default=32,
        help="dim hidden layers",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=4,
        help="num hidden layers",
    )
    parser.add_argument(
        "--n-fc-layers",
        type=int,
        default=1,
        help="num linear layers at each ff block",
    )
    parser.add_argument(
        "--n-out-fc",
        type=int,
        default=1,
        help="num linear layers at final layer (invariant block)",
    )
    parser.add_argument(
        "--set-layer",
        type=str,
        default="sab",
        choices=["sab", "ds"],
        help="set layer",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="number of attention heads",
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default="dataset/statistics.pth",
        help="path to dataset statistics",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="dataset/mnist_splits.json",
        help="path to dataset statistics",
    )
    parser.add_argument("--eval-every", type=int, default=1, help="eval every")
    parser.add_argument(
        "--augmentation", type=str2bool, default=True, help="use augmentation"
    )
    parser.add_argument(
        "--permutation", type=str2bool, default=False, help="use permutations"
    )
    parser.add_argument(
        "--normalize", type=str2bool, default=True, help="normalize data"
    )
    parser.add_argument("--diagonal", type=str2bool, default=False, help="diagonal DWSNet")
    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--add-bn", type=str2bool, default=True, help="add batch norm layers"
    )
    args = parser.parse_args()

    # Store args globally for backward compatibility
    _cli_args = args

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = (
            f"{args.dataset_name}_clf_{args.model}_lr_{args.lr}_hid_dim_{args.dim_hidden}_reduction_{args.reduction}"
            f"_bs_{args.batch_size}_seed_{args.seed}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device(gpus=args.gpu)

    main(
        dataset_name=args.dataset_name,
        epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        device=device,
        base_path=args.base_path,
    )
