import os
import yaml
import torch
import logging
import torch.nn.functional as F
from torch import nn
from typing import Optional
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


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
from utils.wandb_manager import WandbManager

set_logger()


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


def main(conf: dict, wandb_manager: Optional[WandbManager] = None):
    """
    Train DWS (Deep Weight Space) classifier.

    Args:
        conf: Configuration dict with keys:
            - dataset_name: 'mnist', 'fashion', or 'cifar'
            - augmentation, permutation, normalize, diagonal, do_rate, add_bn
            - statistics_path, split_path, base_path, siren_path
            - model_type, dim_hidden, n_hidden, n_fc_layers, reduction, n_out_fc, set_layer, n_heads
            - n_samples, n_epochs, eval_every
            - optim, lr, batch_size, num_workers
            - seed, gpu, save_path
            - wandb: Whether to log to wandb
        wandb_manager: Optional WandbManager for logging
    """
    # Extract config values with defaults
    dataset_name = conf.get('dataset_name', 'mnist')
    augmentation = conf.get('augmentation', False)
    permutation = conf.get('permutation', False)
    normalize = conf.get('normalize', False)
    diagonal = conf.get('diagonal', False)
    do_rate = conf.get('do_rate', 0.0)
    add_bn = conf.get('add_bn', True)

    # Paths
    siren_path = conf.get('siren_path', None)
    inr_type = conf.get('inr_type', 'siren')  # 'siren' or 'ffn'
    prefix = conf.get('prefix', 'randinit_smaller')
    extra_aug = conf.get('extra_aug', False)

    # Derive statistics_path from siren_path if provided
    # FFN uses different statistics file because first layer is skipped (different weight shapes)

    inr_suffix = "_ffn" if inr_type == "ffn" else ""
    name = prefix + "_aug" * extra_aug
    if siren_path is not None:
        statistics_path = os.path.join(siren_path, f"{name}_statistics{inr_suffix}.pth")
    else:
        statistics_path = conf.get('statistics_path', f'dataset/{name}_statistics{inr_suffix}.pth')

    # Model options
    dim_hidden = conf.get('dim_hidden', 32)
    n_hidden = conf.get('n_hidden', 4)
    n_fc_layers = conf.get('n_fc_layers', 1)
    n_out_fc = conf.get('n_out_fc', 1)

    model_type = conf.get('model_type', 'dwsnet')
    reduction = conf.get('reduction', 'max')
    set_layer = conf.get('set_layer', 'sab')
    n_heads = conf.get('n_heads', 8)

    # Training options
    n_samples = conf.get('n_samples', None)
    n_epochs = conf.get('n_epochs', 100)
    eval_every = conf.get('eval_every', 1)

    # Optimizer options
    optim = conf.get('optim', 'adamw')
    lr = conf.get('lr', 1e-2)
    batch_size = conf.get('batch_size', 512)
    num_workers = conf.get('num_workers', 8)

    # General
    seed = conf.get('seed', 42)
    gpu = conf.get('gpu', 0)
    save_path = conf.get('save_path', None)

    # Setup device
    device = get_device(gpus=str(gpu))

    # Set seed
    set_seed(seed)

    # Initialize wandb manager if not provided
    owns_wandb = wandb_manager is None
    if owns_wandb:
        wandb_enabled = conf.get('wandb', False)
        wandb_manager = WandbManager(enabled=wandb_enabled)

    # Initialize wandb run
    if wandb_manager.enabled and wandb_manager.current_run is None:
        run_name = wandb_manager.generate_run_name(
            dataset=dataset_name,
            model="dws",
            training_type="classifier",
        )
        wandb_manager.init_run(
            name=run_name,
            group=f"dws_{dataset_name}",
            tags=wandb_manager.generate_tags(
                dataset=dataset_name,
                model="dws",
                training_type="classifier",
            ),
            config=conf,
            job_type="training",
        )

    # Load dataset - both SIREN and FFN use splits.json format
    if siren_path is None:
        raise ValueError("siren_path must be provided for DWS training")
    if not os.path.isdir(siren_path):
        raise ValueError(f"siren_path must be a directory, got: {siren_path}")

    # load dataset - INRDataset expects path to splits.json file, not directory
    splits_json_path = os.path.join(siren_path, f"{name}_splits.json")
    if not os.path.isfile(splits_json_path):
        raise FileNotFoundError(
            f"splits.json not found at {splits_json_path}. "
            f"Ensure _ensure_dws_splits_json was called before training."
        )
    if not os.path.isfile(statistics_path):
        raise FileNotFoundError(
            f"statistics.pth not found at {statistics_path}. "
            f"Ensure _ensure_dws_statistics was called before training."
        )

    train_set = INRDataset(
        path=splits_json_path,
        split="train",
        normalize=normalize,
        augmentation=augmentation,
        permutation=permutation,
        statistics_path=statistics_path,
    )
    val_set = INRDataset(
        path=splits_json_path,
        split="val",
        normalize=normalize,
        statistics_path=statistics_path,
    )
    test_set = INRDataset(
        path=splits_json_path,
        split="test",
        normalize=normalize,
        statistics_path=statistics_path,
    )

    # if n_samples is not None:
    #     _, train_indices = train_test_split(
    #         range(len(train_set)), test_size=n_samples
    #     )
    #     train_set = torch.utils.data.Subset(train_set, train_indices)

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

    print(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}, "
        f"batch size {batch_size}"
    )

    point = train_set.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)

    print(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

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


    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    for epoch in range(n_epochs):
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

            wandb_manager.log({
                "train/loss": loss.item(),
            })

            # epoch_iter.set_description(
            #     f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
            # )
            if (i + 1) % 100 == 0 and i > 0:
                print(f"epoch {epoch} iter {i+1}/{len(train_loader)}, train_loss: {loss.item():.4f}")
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
                ckpt_path = os.path.join(save_path, "best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved model to {ckpt_path} at epoch {epoch}")

            wandb_manager.log({
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
                "epoch": epoch,
            })
            print(f"epoch {epoch} | train_loss: {train_loss_dict['avg_loss']} train_acc: {train_loss_dict['avg_acc']}, val_loss: {val_loss} val_acc {val_acc}, test_loss: {test_loss} test_acc: {test_acc}")

if __name__ == "__main__":
    parser = ArgumentParser(
        "INR classification trainer", parents=[common_parser]
    )

    parser.set_defaults(
        n_epochs=100,
        batch_size=512,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config file (YAML)",
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
        "--siren-path",
        type=str,
        default=None,
        help="path to SIREN weights directory",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="path to save checkpoints",
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

    # Load config file if provided, otherwise use defaults
    if args.config:
        with open(args.config) as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}

    # Override with CLI arguments (convert hyphens to underscores for config keys)
    cli_overrides = {
        'dataset_name': args.dataset_name,
        'base_path': args.base_path,
        'model_type': args.model,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'gpu': args.gpu,
        'optim': args.optim,
        'num_workers': args.num_workers,
        'reduction': args.reduction,
        'dim_hidden': args.dim_hidden,
        'n_hidden': args.n_hidden,
        'n_fc_layers': args.n_fc_layers,
        'n_out_fc': args.n_out_fc,
        'set_layer': args.set_layer,
        'n_heads': args.n_heads,
        'statistics_path': args.statistics_path,
        'split_path': args.split_path,
        'eval_every': args.eval_every,
        'augmentation': args.augmentation,
        'permutation': args.permutation,
        'normalize': args.normalize,
        'diagonal': args.diagonal,
        'do_rate': args.do_rate,
        'add_bn': args.add_bn,
        'wandb': args.wandb,
    }

    # Add optional args only if provided
    if args.siren_path:
        cli_overrides['siren_path'] = args.siren_path
    if args.save_path:
        cli_overrides['save_path'] = args.save_path
    if args.n_samples:
        cli_overrides['n_samples'] = args.n_samples

    # Merge config with CLI overrides (CLI takes precedence)
    conf.update(cli_overrides)

    # Create wandb manager
    wandb_manager = WandbManager(enabled=args.wandb)

    # Run training
    main(conf, wandb_manager)
