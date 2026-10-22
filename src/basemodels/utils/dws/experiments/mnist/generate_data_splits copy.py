import json
import logging
import os
import re
import glob
from argparse import ArgumentParser
# from collections import defaultdict
# from pathlib import Path

# from sklearn.model_selection import train_test_split

# from experiments.utils import common_parser, set_logger


def generate_splits(data_path, save_path, name="mnist_splits.json", val_size=5000, dataset_name="mnist"):
    save_path = f"{save_path}/siren_{dataset_name}_wts/{name}"
    inr_path = f"{data_path}/siren_{dataset_name}_wts"

    prefix = "randinit_smaller"
    idx_pattern = r"net(\d+)\.pth"
    label_pattern = r"_(\d)s"
    idx_to_path, idx_to_label = {}, {}

    for siren_path in glob.glob(f"{inr_path}/{prefix}_[0-9]s/*.pth"):
        idx = int(re.search(idx_pattern, siren_path).group(1))
        idx_to_path[idx] = siren_path
        label = int(re.search(label_pattern, siren_path).group(1))
        idx_to_label[idx] = label

    val_point = 45000 if dataset_name == "cifar" else 55000
    test_point = 55000 if dataset_name == "cifar" else 60000

    idx = {
        "train": list(range(val_point)),
        "val": list(range(val_point, test_point)),
        "test": list(range(test_point, len(idx_to_path))),
    }
    # total 70000 for mnist and fashion, 60000 for cifar

    # if not aug: 
    # train: 0-44999 (0-54999 for mnist),
    # val: 45000-54999 (55000-59999 for mnist),
    # test: 55000-60000 (60000-70000 for mnist)

    # if aug:
    # train: 0-44999, 65000-65000+45000 * num_aug
    # val: 45000-54999
    # test: 55000-60000


    data_split={"train": {"path": [], "label": []}, "val": {"path": [], "label": []}, "test": {"path": [], "label": []}}
    data_split['train']['path'] = [idx_to_path[i] for i in idx['train']]
    data_split['val']['path'] = [idx_to_path[i] for i in idx['val']]
    data_split['test']['path'] = [idx_to_path[i] for i in idx['test']]
    
    data_split['train']['label'] = [idx_to_label[i] for i in idx['train']]
    data_split['val']['label'] = [idx_to_label[i] for i in idx['val']]
    data_split['test']['label'] = [idx_to_label[i] for i in idx['test']]
    # for p in list(inr_path.glob("mnist_png_*/**/*.pth")):
    #     if "train" in p.as_posix():
    #         s = "train"
    #     else:
    #         s = "test"
    #     data_split[s]["path"].append((os.getcwd() / p).as_posix())
    #     data_split[s]["label"].append(p.parent.parent.stem.split("_")[-2])

    # # val split
    # val_size = val_size
    # train_indices, val_indices = train_test_split(
    #     range(len(data_split["train"]["path"])), test_size=val_size
    # )
    # data_split["val"]["path"] = [data_split["train"]["path"][v] for v in val_indices]
    # data_split["val"]["label"] = [data_split["train"]["label"][v] for v in val_indices]

    # data_split["train"]["path"] = [
    #     data_split["train"]["path"][v] for v in train_indices
    # ]
    # data_split["train"]["label"] = [
    #     data_split["train"]["label"][v] for v in train_indices
    # ]
    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, test size: {len(data_split['test']['path'])}"
    )

    with open(save_path, "w") as file:
        json.dump(data_split, file)


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate data splits")
    # parser = ArgumentParser("MNIST - generate data splits", parents=[common_parser])
    parser.add_argument(
        "--dataset_name", type=str, default="fashion", help="dataset name"
    )
    parser.add_argument(
        "--name", type=str, default="fashion_splits.json", help="json file name"
    )
    parser.add_argument(
        "--val-size", type=int, default=5000, help="number of validation examples"
    )
    parser.set_defaults(
        save_path="/nfs/stak/users/hongsa/INR-Robustness/sirens",
        data_path="/nfs/stak/users/hongsa/INR-Robustness/sirens",
    )
    
    args = parser.parse_args()

    # set_logger()

    generate_splits(
        data_path=args.data_path,
        save_path=args.save_path,
        name=args.name,
        val_size=args.val_size,
        dataset_name=args.dataset_name,
    )
