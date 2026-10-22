from argparse import ArgumentParser

from torchvision.utils import save_image, make_grid
import torch

from basemodels.utils.dws.experiments.data import INRImageDataset
from basemodels.utils.dws.experiments.utils import common_parser


parser = ArgumentParser(
    "MNIST - Reconstruct images grid from INRs", parents=[common_parser]
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="cifar",
    choices=["mnist", "cifar", "fashion"],
    help="dataset_name",
)
args = parser.parse_args()

dataset = INRImageDataset(
    path=f"/nfs/stak/users/hongsa/INR-Robustness/sirens/siren_{args.dataset_name}_wts/splits.json",  # path to splits json file
    augmentation=False,
    split="train",
    inr_kwargs={"n_layers": 3, "in_dim": 2, "up_scale": 16, "out_channels": 3 if args.dataset_name == "cifar" else 1},
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
batch = next(iter(loader))
save_image(batch.image.squeeze(-1), "mnist_grid.png")
