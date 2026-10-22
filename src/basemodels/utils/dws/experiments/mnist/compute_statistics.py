from argparse import ArgumentParser
from pathlib import Path

import torch

from basemodels.utils.dws.experiments.data import Batch, INRDataset
from basemodels.utils.dws.experiments.utils import common_parser


def compute_stats(data_path: str, save_path: str, batch_size: int = 10000):
    train_set = INRDataset(path=data_path, split="train", normalize=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )

    batch: Batch = next(iter(train_loader))
    weights_mean = [w.mean(0) for w in batch.weights]
    weights_std = [w.std(0) for w in batch.weights]
    biases_mean = [w.mean(0) for w in batch.biases]
    biases_std = [w.std(0) for w in batch.biases]

    statistics = {
        "weights": {"mean": weights_mean, "std": weights_std},
        "biases": {"mean": biases_mean, "std": biases_std},
    }

    out_path = Path(save_path)
    # out_path.mkdir(exist_ok=True, parents=True)

    file_name = "statistics.pth"
    torch.save(statistics, out_path / file_name)


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate statistics", parents=[common_parser])
    parser.set_defaults(batch_size=10000, 
                        dataset_name="cifar")
    args = parser.parse_args()

    save_path = f"/nfs/stak/users/hongsa/INR-Robustness/sirens/siren_{args.dataset_name}_wts"
    data_path = f"/nfs/stak/users/hongsa/INR-Robustness/sirens/siren_{args.dataset_name}_wts/splits.json"
    # save_path = f"/nfs/stak/users/hongsa/INR-Robustness/siren-from-dws/mnist-inrs"
    # data_path = f"/nfs/stak/users/hongsa/INR-Robustness/siren-from-dws/mnist-inrs/mnist_splits.json"

    compute_stats(
        data_path=data_path, save_path=save_path, batch_size=args.batch_size
    )
