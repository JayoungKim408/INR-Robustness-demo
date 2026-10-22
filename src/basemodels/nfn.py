"""
NFN (Neural Functional Networks) Training Script
Unified YAML config - no OmegaConf/Hydra dependencies
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
import time
import numpy as np
import torch.multiprocessing
import torch.nn.functional as F
from functools import partial
from typing import Optional, Union, Dict
from torch.utils import data
from datetime import timedelta
from torch.utils.data import ConcatDataset

from attacks.utils import BatchedINRBuilder
from basemodels.utils.nfn.train_utils import set_seed
from basemodels.utils.nfn.data_utils import get_SirenDataset
from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from basemodels.utils.nfn.data_utils import compute_mean_std, cycle
from basemodels.utils.nfn.models import InvariantNFN
from utils.wandb_manager import WandbManager
from utils.config_utils import DictConfig, config_to_dict


class SimpleTimer:
    def __init__(self, total_steps):
        self.total = total_steps
        self.start = time.time()

    def get_eta(self, current_step):
        elapsed = time.time() - self.start
        speed = current_step / elapsed if elapsed > 0 else 0
        eta = (self.total - current_step) / speed if speed > 0 else 0
        return str(timedelta(seconds=int(eta)))


def _get_cfg_value(cfg, key, default=None):
    """Get value from config (supports both dict and DictConfig)."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    elif hasattr(cfg, key):
        return getattr(cfg, key)
    else:
        return default


def evaluate(nfnet, loader):
    orig_state = nfnet.training
    nfnet.eval()
    # pbar = tqdm(loader, position=1, leave=False)
    labels, preds = [], []
    val_losses = []
    for wts_and_bs, _, label in loader:
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        with torch.no_grad():
            pred = nfnet(params)
        loss = F.cross_entropy(pred, label.cuda())
        # pbar.set_description(f"val loss={loss.item():.3f}")
        preds.append(torch.argmax(pred, -1).cpu().numpy())
        labels.append(label.numpy())
        val_losses.append(loss.item())
    # pbar.close()
    val_acc = (np.concatenate(preds) == np.concatenate(labels)).mean().item()
    nfnet.train(orig_state)
    return val_acc, np.mean(val_losses)


def train_step(nfnet, params, label):
    # opt.zero_grad(True)
    pred = nfnet(params)
    loss = F.cross_entropy(pred, label.cuda())
    # loss.backward()
    # opt.step()
    return loss, pred


def main(cfg: Union[Dict, DictConfig], 
         wandb_manager: Optional[WandbManager] = None):
    """
    Train NFN classifier.

    Args:
        cfg: Configuration dict or DictConfig with keys:
            - data_name: dataset name ('mnist', 'fashion', 'cifar')
            - batch_size, warmup_steps, max_steps, seed, compile, extra_aug
            - siren_clf_dir: save directory
            - dset: dataset config (siren_path, prefix, split_points)
            - opt: optimizer config (optimizer, lr)
            - nfnet: model config (hchannels, mode, etc.)
        wandb_manager: Optional WandbManager for logging
    """
    # Convert to DictConfig if needed for consistent access
    if isinstance(cfg, dict) and not isinstance(cfg, DictConfig):
        cfg = DictConfig(cfg)

    set_seed(_get_cfg_value(cfg, 'seed', 42))

    # Initialize wandb manager if not provided
    owns_wandb = wandb_manager is None
    if owns_wandb:
        wandb_enabled = _get_cfg_value(cfg, 'wandb', False)
        wandb_manager = WandbManager(enabled=wandb_enabled)

    prev_ckpt, wandb_id = None, None
    ckpt_path = os.path.join(cfg.siren_clf_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        prev_ckpt = torch.load(ckpt_path, weights_only=False)
        wandb_id = prev_ckpt.get("wandb_run_id")

    # Initialize wandb run
    if wandb_manager.enabled and wandb_manager.current_run is None:
        run_name = wandb_manager.generate_run_name(
            dataset=cfg.data_name,
            model="nfn",
            training_type="classify_siren",
        )
        wandb_manager.init_run(
            name=run_name,
            group=f"classify_siren_{cfg.data_name}",
            tags=wandb_manager.generate_tags(
                dataset=cfg.data_name,
                model="nfn",
                training_type="classify_siren",
            ),
            config=config_to_dict(cfg) if prev_ckpt is None else None,
            run_id=wandb_id,
            resume="must" if wandb_id else None,
            job_type="training",
        )

    # Create datasets directly (no hydra.utils.instantiate)
    dset_cfg = cfg.dset if hasattr(cfg, 'dset') else cfg.get('dset', {})
    # siren_path = dset_cfg.get('siren_path') or dset_cfg.get('data_path')
    # siren_prefix = dset_cfg.get('prefix', 'init_pool')
    # split_points = dset_cfg.get('split_points', [45000, 50000])
    # inr_type = dset_cfg.get('inr_type', 'siren')

    trainset = get_SirenDataset(**dset_cfg, split="train", )
    # from torch.utils.data import Subset
    # trainset = Subset(trainset, range(100))

    valset = get_SirenDataset(**dset_cfg, split="val", )

    extra_aug = _get_cfg_value(cfg, 'extra_aug', True)
    n_extra_aug = _get_cfg_value(cfg, 'n_extra_aug', 10)

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

    batch_size = _get_cfg_value(cfg, 'batch_size', 32)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    valloader = data.DataLoader(valset, batch_size=min(batch_size, len(valset)), num_workers=8)
    print(f"Dataset sizes: train={len(trainset)}, val={len(valset)}, batch_size={batch_size}.")

    sample = next(iter(trainloader))
    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*sample[0]).to("cpu"), set_all_dims=True)

    # Create model directly (no hydra.utils.instantiate)
    nfnet_cfg = cfg.nfnet if hasattr(cfg, 'nfnet') else cfg.get('nfnet', {})
    if isinstance(nfnet_cfg, DictConfig):
        nfnet_cfg = nfnet_cfg.to_dict()

    # Extract model params from config
    hchannels = nfnet_cfg.get('hchannels', [512, 512, 512])
    mode = nfnet_cfg.get('mode', 'NP')
    feature_dropout = nfnet_cfg.get('feature_dropout', 0.0)
    normalize = nfnet_cfg.get('normalize', False)
    lnorm = nfnet_cfg.get('lnorm', None)
    append_stats = nfnet_cfg.get('append_stats', False)

    # Create head_cls factory from head config
    head_cfg = nfnet_cfg.get('head', {})
    from basemodels.utils.nfn.models import MlpHead
    head_cls = partial(
        MlpHead,
        num_out=head_cfg.get('num_out', 10),
        dropout=head_cfg.get('dropout', 0.5),
        lnorm=head_cfg.get('lnorm', False),
        sigmoid=head_cfg.get('sigmoid', False),
        pool_mode=mode,
    )

    # Create inp_enc_cls factory from inp_enc config
    inp_enc_cfg = nfnet_cfg.get('inp_enc', {})
    from basemodels.utils.nfn.nfn.layers import GaussianFourierFeatureTransform
    inp_enc_cls = partial(
        GaussianFourierFeatureTransform,
        mapping_size=inp_enc_cfg.get('mapping_size', 128),
        scale=inp_enc_cfg.get('scale', 10),
    ) if inp_enc_cfg else None

    # Create pos_enc_cls factory from pos_enc config
    pos_enc_cfg = nfnet_cfg.get('pos_enc', {})
    from basemodels.utils.nfn.nfn.layers import IOSinusoidalEncoding
    pos_enc_cls = partial(
        IOSinusoidalEncoding,
        max_freq=pos_enc_cfg.get('max_freq', 10),
        num_bands=pos_enc_cfg.get('num_bands', 6),
    ) if pos_enc_cfg else None

    nfnet = InvariantNFN(
        network_spec=spec,
        hchannels=hchannels,
        head_cls=head_cls,
        mode=mode,
        feature_dropout=feature_dropout,
        normalize=normalize,
        lnorm=lnorm,
        append_stats=append_stats,
        inp_enc_cls=inp_enc_cls,
        pos_enc_cls=pos_enc_cls,
    )
    print(f"Total params in NFN: {sum(p.numel() for p in nfnet.parameters()) / 1000000}M.")

    is_3d = cfg.data_name == 'shapenet'
    if not is_3d:
        builder = BatchedINRBuilder(
            sidelen=32 if cfg.data_name == "cifar" else 28,
            hidden_features=cfg.hidden_features,
            hidden_layers=cfg.hidden_layers,
            c_out=3 if cfg.data_name == "cifar" else 1,
            first_omega_0=cfg.first_omega_0,
            hidden_omega_0=cfg.hidden_omega_0,
            device="cuda",
        )

    if nfnet.normalize:
        nfnet.set_stats(compute_mean_std(trainloader, max_batches=(5_000 // batch_size) + 1))
    print(nfnet)
    nfnet.cuda()

    # accum_steps = 4 if is_3d else 1
    accum_steps = _get_cfg_value(cfg, 'accum_steps', 4 if is_3d else 1)

    compile_model = _get_cfg_value(cfg, 'compile', True)
    if compile_model and not is_3d:
        nfnet = torch.compile(nfnet)

    # Create optimizer directly
    opt_cfg = cfg.opt if hasattr(cfg, 'opt') else cfg.get('opt', {})
    lr = opt_cfg.get('lr', 1e-4) if isinstance(opt_cfg, dict) else getattr(opt_cfg, 'lr', 1e-4)
    opt = torch.optim.Adam(nfnet.parameters(), lr=lr)

    sched = None
    warmup_steps = _get_cfg_value(cfg, 'warmup_steps', 0) // accum_steps
    if warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / warmup_steps))
    start_step, best_val_acc = 0, 0

    if os.path.exists(ckpt_path):
        nfnet.load_state_dict(prev_ckpt["nfnet"])
        opt.load_state_dict(prev_ckpt["opt"])
        start_step = prev_ckpt["step"]
        best_val_acc = prev_ckpt["best_val_acc"]
        if sched is not None:   
            print("sched loaded")
            sched.load_state_dict(prev_ckpt["sched"])

    from datetime import datetime

    start_time = datetime.now()
    train_iter = cycle(trainloader)
    max_steps = _get_cfg_value(cfg, 'max_steps', 200000)
    # outer_pbar = trange(start_step, max_steps, position=0)
    timer = SimpleTimer(max_steps)

    print(f"Starting training: {start_step} to {max_steps}")
    val_acc = 0.0
    opt.zero_grad()
    for step in range(start_step, max_steps):
        log_metric = {}
        if step % 100 == 0 or step == max_steps - 1:
            val_acc, val_loss = evaluate(nfnet, valloader)
            val_sample = next(iter(valloader))
            if not is_3d:
                imgs = builder.build_img_from_wts_and_bs(val_sample[0])
                imgs = (imgs + 1) / 2
                # if wandb_manager.is_active:
                #     wandb_manager.log({"val/imgs": [wandb.Image(img) for img in imgs.detach().cpu().numpy()]})
                log_metric["val/imgs"] = [wandb.Image(img) for img in imgs.detach().cpu().numpy()]
            log_metric["val/acc"] = val_acc
            log_metric["val/loss"] = val_loss
            # wandb_manager.log({"val/acc": val_acc, "val/loss": val_loss}, step=step)
            # print(f"iteration: {step}, val_acc={val_acc:.3f}", end="\r", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(nfnet.state_dict(), os.path.join(cfg.siren_clf_dir, "best.pt"))
            torch.save({
                "step": step,
                "nfnet": nfnet.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched else None,
                "best_val_acc": best_val_acc,
                "wandb_run_id": wandb_manager.get_run_id(),
            }, ckpt_path)

        wts_and_bs, _, label = next(train_iter)
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")

        loss, pred = train_step(nfnet, params, label)
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            opt.step()
            opt.zero_grad()

            if sched is not None:
                sched.step()

        # if step % 10 == 0:
        train_acc = (torch.argmax(pred.detach(), -1).cpu().numpy() == label.numpy()).mean().item()
        log_metric['train/loss'] = loss.item()
        log_metric["train/acc"] = train_acc
        if sched is not None:
            log_metric["train/lr"] = sched.get_last_lr()[0]
        wandb_manager.log(log_metric, step=step)
        if step % 1000 == 0:
            print(f"iteration: {step}, train_loss={loss.item():.3f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, eta={timer.get_eta(step)}, lr={sched.get_last_lr()[0]}", end="\r", flush=True)
    # outer_pbar.close()

    # Load best and evaluate on test
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Training time elapsed: {elapsed}")

    nfnet.load_state_dict(torch.load(os.path.join(cfg.siren_clf_dir, "best.pt"), weights_only=False))
    testset = get_SirenDataset(**dset_cfg, split="test",)
    print(f"Test set with {len(testset)} examples.")
    testloader = data.DataLoader(testset, batch_size=batch_size, num_workers=8)
    test_acc, test_loss = evaluate(nfnet, testloader)
    print(f"Test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}.")
    wandb_manager.log({"test/acc": test_acc, "test/loss": test_loss}, step=step)

    final_train_acc, final_train_loss = evaluate(nfnet, trainloader)
    print(f"Final train accuracy: {final_train_acc:.3f}, final train loss: {final_train_loss:.3f} | {os.path.join(cfg.siren_clf_dir, 'best.pt')}")
    wandb_manager.log({"final_train/acc": final_train_acc, "final_train/loss": final_train_loss}, step=step)

    # Log summary metrics
    wandb_manager.log_summary({
        "test/acc": test_acc,
        "test/loss": test_loss,
        "final_train/acc": final_train_acc,
        "final_train/loss": final_train_loss,
        "best_val_acc": best_val_acc,
    })

    # Only finish if we own the wandb manager
    if owns_wandb:
        wandb_manager.finish_run()
