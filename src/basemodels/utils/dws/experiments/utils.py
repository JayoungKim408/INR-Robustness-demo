import argparse
import logging
import random
from typing import List, Tuple, Union

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _add_common_arguments(parser):
    """Add common arguments to a parser."""
    # Dataset
    parser.add_argument("--dataset_name", type=str, default="cifar", choices=["mnist", "cifar", "fashion"], help="dataset_name",)
    parser.add_argument("--augmentation", type=str2bool, default=False, help="use augmentation")
    parser.add_argument("--permutation", type=str2bool, default=False, help="use permutations")
    parser.add_argument("--normalize", type=str2bool, default=False, help="normalize data")
    parser.add_argument("--diagonal", type=str2bool, default=False, help="diagonal DWSNet")
    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--add-bn", type=str2bool, default=True, help="add batch norm layers")

    # Paths
    parser.add_argument("--statistics-path", type=str, default="dataset/statistics.pth", help="path to dataset statistics")
    parser.add_argument("--split-path", type=str, default="dataset/mnist_splits.json", help="path to dataset statistics")
    parser.add_argument("--data-path", type=str, help="path for dataset")
    parser.add_argument("--base-path", type=str, default="/nfs/stak/users/hongsa/INR-Robustness", help="base path for data and checkpoints",)

    # Model
    parser.add_argument("--model_type", type=str, default="dwsnet", choices=["dwsnet", "mlp"], help="model",)
    parser.add_argument("--dim-hidden", type=int, default=32, help="dim hidden layers",)
    parser.add_argument("--n-hidden", type=int, default=4, help="num hidden layers",)
    parser.add_argument("--n-fc-layers", type=int, default=1, help="num linear layers at each ff block",)
    parser.add_argument("--reduction", type=str, default="max", choices=["mean", "sum", "max"], help="reduction strategy")
    parser.add_argument("--n-out-fc", type=int, default=1, help="num linear layers at final layer (invariant block)",)
    parser.add_argument("--set-layer", type=str, default="sab", choices=["sab", "ds"], help="set layer",)
    parser.add_argument("--n-heads", type=int, default=8, help="number of attention heads")

    # Training
    parser.add_argument("--n-samples", type=int, default=None, help="num training samples")
    parser.add_argument("--n-epochs", type=int, default=100, help="num epochs")
    parser.add_argument("--eval-every", type=int, default=1, help="eval every")

    # Optimization
    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "sgd", "adamw"], help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="inner learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="num workers")

    # General
    parser.add_argument("--seed", type=int, default=42, help="seed value for 'set_seed'")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--save-path", type=str, default="./output", help="dir path for output file")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="dws-nets", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, help="wandb entity name")
    parser.add_argument("--wandb", dest="wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=False, n_epochs=100, batch_size=512)

    return parser


# Create a module-level common_parser for use as a parent parser
common_parser = argparse.ArgumentParser(add_help=False, description="common parser")
_add_common_arguments(common_parser)


def get_parser():
    """Get parser and parse arguments."""
    parser = argparse.ArgumentParser(add_help=False, description="common parser")
    _add_common_arguments(parser)
    args = parser.parse_args()
    return args


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
