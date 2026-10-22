"""
INR2Array (AutoEncoder) Training Script
Unified YAML config - no OmegaConf/Hydra dependencies
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import wandb
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from PIL import Image
from torch import nn
from functools import partial
from datetime import timedelta
from typing import Optional, Union, Dict
from einops.layers.torch import Reduce, Rearrange
from torch.utils.data import DataLoader, ConcatDataset

from basemodels.utils.nfn.nfn.layers import *
from basemodels.utils.nfn.siren_utils import *
from basemodels.utils.nfn.data_utils import *
from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures, NetworkSpec, params_to_func_params
from basemodels.utils.nfn.train_utils import get_linear_warmup_with_cos_decay, set_seed
from basemodels.utils.nfn.models import Block
from utils.wandb_manager import WandbManager
from utils.config_utils import DictConfig, config_to_dict
from perceiver_pytorch import Perceiver
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    elif isinstance(cfg, dict):
        return cfg.get(key, default)
    elif hasattr(cfg, key):
        return getattr(cfg, key)
    else:
        return default


def simple_npblock(network_spec, n_chan):
    return nn.Sequential(NPLinear(network_spec, n_chan, n_chan), TupleOp(nn.ReLU()))


def np_resblock(network_spec, n_chan, dropout=0.1):
    return ResBlock(
        NPLinear(network_spec, n_chan, n_chan),
        TupleOp(nn.ReLU()), dropout=dropout,
        norm=SimpleLayerNorm(network_spec, n_chan),
    )


def hnp_resblock(network_spec, n_chan, dropout=0.1):
    return ResBlock(
        HNPLinear(network_spec, n_chan, n_chan),
        TupleOp(nn.ReLU()), dropout=dropout,
        norm=SimpleLayerNorm(network_spec, n_chan),
    )


class PerceiverPooling(nn.Module):
    def __init__(self, network_spec, n_chan, n_latent=64, latent_dim=256, reduce=True, attn_dropout=0.1, ff_dropout=0.1, self_per_cross_attn=2, num_freq_bands=6, max_freq=10.0):
        super().__init__()
        self.reduce, self.n_latent = reduce, n_latent
        self.net = nn.Sequential(
            FlattenWeights(network_spec),
            Perceiver(
                input_channels=n_chan,
                input_axis=1,
                depth=1,
                num_latents=n_latent,
                num_freq_bands=6,
                max_freq=10,
                latent_dim=latent_dim,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=n_chan,
                latent_dim_head=64,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                weight_tie_layers=False,
                fourier_encode_data=False,
                self_per_cross_attn=self_per_cross_attn,
                final_classifier_head=False,
            ),
            Reduce("b n c -> b c", reduction="mean") if reduce else nn.Identity(),
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.net(x)

    def get_feature_shape(self):
        return (self.latent_dim,) if self.reduce else (self.n_latent, self.latent_dim)


class NPPooling(nn.Module):
    def __init__(self, network_spec, n_chan, reduce=True, num_patches=64, latent_dim=64):
        super().__init__()
        self.reduce, self.latent_dim = reduce, latent_dim
        self.n_chan, self.num_patches = n_chan, num_patches
        num_pool_outs = NPPool.get_num_outs(network_spec)
        num_latent = latent_dim if reduce else num_patches * latent_dim
        self.pool = nn.Sequential(
            NPPool(network_spec), nn.Flatten(-2),
            nn.Linear(num_pool_outs * n_chan, 1000), nn.ReLU(),
            nn.Linear(1000, num_latent)
        )
        self.reduce_fn = nn.Identity()
        if not reduce:
            self.reduce_fn = Rearrange("b (l d) -> b l d", l=num_patches, d=latent_dim)

    def forward(self, x):
        return self.reduce_fn(self.pool(x))

    def get_feature_shape(self):
        return (self.latent_dim,) if self.reduce else (self.num_patches, self.latent_dim)


BLOCK_TYPES = {
    "np": simple_npblock,
    "np_residual": np_resblock,
    "hnp_residual": hnp_resblock,
    "nft": Block,
}
DEC_TYPES = {"sitzmann": HyperNetwork, "simple": SimpleHyperNetwork}


class AutoEncoder(nn.Module):
    def __init__(
        self,
        model_cfg: Union[Dict, DictConfig],
        network_spec: NetworkSpec,
        dset_data_type,
        pool_cls=None,
    ):
        super().__init__()
        # Extract config values
        if isinstance(model_cfg, DictConfig):
            model_cfg = model_cfg.to_dict()

        block_type = model_cfg.get('block_type', 'nft')
        num_blocks = model_cfg.get('num_blocks', 3)
        spatial = model_cfg.get('spatial', False)
        compile_model = model_cfg.get('compile', False)
        enc_scale = model_cfg.get('enc_scale', 3)
        enc_map_size = model_cfg.get('enc_map_size', 128)
        dec_type = model_cfg.get('dec_type', 'sitzmann')
        dec_hidden_size = model_cfg.get('dec_hidden_size', 256)
        dec_hidden_layers = model_cfg.get('dec_hidden_layers', 1)
        additive = model_cfg.get('additive', False)
        debug_compile = model_cfg.get('debug_compile', False)

        # Block-specific kwargs
        block_kwargs = {}
        if 'num_heads' in model_cfg:
            block_kwargs['num_heads'] = model_cfg['num_heads']
        if 'ff_factor' in model_cfg:
            block_kwargs['ff_factor'] = model_cfg['ff_factor']
        if 'dropout' in model_cfg:
            block_kwargs['dropout'] = model_cfg['dropout']

        self.network_spec = network_spec
        self.spatial = spatial
        self.compile, self.debug_compile = compile_model, debug_compile
        n_chan = 2 * enc_map_size

        # Use provided pool_cls or default to PerceiverPooling
        if pool_cls is None:
            pool_cls = partial(PerceiverPooling, reduce=not spatial)

        self.encoder = nn.Sequential(
            GaussianFourierFeatureTransform(network_spec, 1, mapping_size=enc_map_size, scale=enc_scale),
            LearnedPosEmbedding(network_spec, n_chan),
            *[BLOCK_TYPES[block_type](network_spec, n_chan, **block_kwargs) for _ in range(num_blocks)],
            pool_cls(network_spec, n_chan),
        )
        self.latent_shape = self.encoder[-1].get_feature_shape()
        self.decoder = nn.Sequential(
            DEC_TYPES[dec_type](network_spec, self.latent_shape[-1], dec_hidden_size, dec_hidden_layers),
            Rearrange("b l -> b l ()"),
            UnflattenWeights(network_spec),
        )
        inr_config = model_cfg.get('inr_config', None)

        if spatial:
            assert len(self.latent_shape) == 2
            # Spatial rendering only supports SIREN for now
            self.batch_siren, init_params = get_spatial_batch_inr(dset_data_type, inr_config=inr_config)
        else:
            # Use unified get_batch_inr for both SIREN and FFN
            self.batch_siren, init_params = get_batch_inr(dset_data_type, inr_config=inr_config)
        self.init_params = None
        if additive:
            self.init_params = nn.ParameterList([nn.Parameter(p) for p in init_params])
            additive_scales = []
            for p in self.init_params:
                if p.numel() == 1:
                    additive_scales.append(nn.Parameter(p.detach().abs().mean()))
                else:
                    additive_scales.append(nn.Parameter(p.detach().std().requires_grad_()))
            self.additive_scales = nn.ParameterList(additive_scales)
        if self.compile:
            self._fast_forward = torch.compile(self._forward_helper)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def _forward_helper(self, x):
        z = self.encoder(x)
        if self.spatial:
            bs, num_latents = z.shape[:2]
            z = torch.flatten(z, end_dim=1)
        out = self.decoder(z)
        if self.spatial:
            def unflatten(arr):
                return arr.reshape(bs, num_latents, *arr.shape[1:]).squeeze(2)
            out = out.map(unflatten)
        return out

    def forward(self, x):
        if self.debug_compile:
            _, _, _, _, _, explanation_verbose = dynamo.explain(self._forward_helper, x)
            print(explanation_verbose)
            self.debug_compile = False
        out = self._fast_forward(x) if self.compile else self._forward_helper(x)
        out = params_to_func_params(out)
        if self.init_params is not None:
            out = [p * scale + init_p[None, None] for p, init_p, scale in zip(out, self.init_params, self.additive_scales)]
        out = self.batch_siren(out)
        return out

    def embedding_to_out(self, x):
        if self.spatial:
            bs, num_latents = x.shape[:2]
            x = torch.flatten(x, end_dim=1)
        out = self.decoder(x)
        if self.spatial:
            def unflatten(arr):
                return arr.reshape(bs, num_latents, *arr.shape[1:]).squeeze(2)
            out = out.map(unflatten)
        out = params_to_func_params(out)
        if self.init_params is not None:
            out = [p * scale + init_p[None, None] for p, init_p, scale in zip(out, self.init_params, self.additive_scales)]
        out = self.batch_siren(out)
        return out

    def sample(self, num_samples):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, *self.latent_shape).to(device)
        if self.spatial:
            _, num_latents = z.shape[:2]
            z = torch.flatten(z, end_dim=1)
        out = self.decoder(z)
        if self.spatial:
            def unflatten(arr):
                return arr.reshape(num_samples, num_latents, *arr.shape[1:]).squeeze(2)
            out = out.map(unflatten)
        out = params_to_func_params(out)
        if self.init_params is not None:
            out = [p * scale + init_p[None, None] for p, init_p, scale in zip(out, self.init_params, self.additive_scales)]
        out = self.batch_siren(out)
        return out


@torch.no_grad()
def evaluate(nfnet, loader, orig_batch_siren, true_target, loss_type='l2', device="cuda"):
    if loader is None:
        raise ValueError("Validation loader is required for evaluation.")
    orig_state = nfnet.training
    nfnet.eval()
    total_loss = torch.zeros(1, device=device)
    for wts_and_bs, init_wts_and_bs, img, _ in loader:
        params = WeightSpaceFeatures(*wts_and_bs).to(device)
        if not true_target:
            img = orig_batch_siren(params_to_func_params(params))
        img = img.to(device)
        pred_img = nfnet(params)
        batch_loss = compute_losses(img, pred_img, loss_type)
        total_loss += batch_loss.detach()
    nfnet.train(orig_state)
    return total_loss / len(loader)


@torch.no_grad()
def sample(nfnet):
    orig_state = nfnet.training
    nfnet.eval()
    samples = nfnet.sample(32)
    nfnet.train(orig_state)
    return unprocess_img_arr(samples.cpu().numpy())


@torch.no_grad()
def sample_recon(nfnet, loader, orig_batch_siren, device="cuda"):
    if loader is None:
        raise ValueError("Loader is required for reconstruction sampling.")
    orig_state = nfnet.training
    nfnet.eval()
    wts_and_bs, init_wts_and_bs, true_img, _ = next(iter(loader))
    params = WeightSpaceFeatures(*wts_and_bs).to(device)
    params, true_img = params.map(lambda x: x[:32]), true_img[:32]
    orig_outs = orig_batch_siren(params_to_func_params(params))
    orig_outs = unprocess_img_arr(orig_outs.cpu().numpy())
    new_outs = nfnet(params)
    new_outs = unprocess_img_arr(new_outs.cpu().numpy())
    nfnet.train(orig_state)
    return orig_outs, new_outs, unprocess_img_arr(true_img.cpu().numpy())


def compute_losses(x, x_recon, loss_type='l2'):
    if loss_type == 'l2':
        return F.mse_loss(x_recon, x)
    else:
        return F.l1_loss(x_recon, x)


def train_and_eval(cfg: Union[Dict, DictConfig], *, use_ddp=False, rank=0, world_size=1, local_rank=0, device=None, wandb_manager: Optional[WandbManager] = None):
    """
    Train INR2Array (AutoEncoder).

    Args:
        cfg: Configuration dict or DictConfig
        wandb_manager: Optional WandbManager for logging
    """
    use_ddp = dist.is_initialized()
    is_main = not use_ddp or rank == 0
    
    # Convert to DictConfig if needed
    if isinstance(cfg, dict) and not isinstance(cfg, DictConfig):
        cfg = DictConfig(cfg)

    set_seed(_get_cfg_value(cfg, 'seed', 42))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb manager if not provided
    owns_wandb = wandb_manager is None
    if owns_wandb:
        debug = _get_cfg_value(cfg, 'debug', False)
        wandb_enabled = not debug and _get_cfg_value(cfg, 'wandb', False)
        wandb_manager = WandbManager(enabled=wandb_enabled)

    ckpt_path = os.path.join(cfg.inr2array_model_dir, "checkpoint.pt")
    print(f"Model will be saved in {ckpt_path}.")
    wandb_id, prev_ckpt = None, None
    if os.path.exists(ckpt_path):
        prev_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        wandb_id = prev_ckpt.get("wandb_run_id")

    # Initialize wandb run
    if wandb_manager.enabled and wandb_manager.current_run is None:
        run_name = wandb_manager.generate_run_name(
            dataset=cfg.dataset_name,
            model="nft",
            training_type="inr2array",
        )
        wandb_manager.init_run(
            name=run_name,
            group=f"inr2array_{cfg.dataset_name}",
            tags=wandb_manager.generate_tags(
                dataset=cfg.dataset_name,
                model="nft",
                training_type="inr2array",
            ),
            config=config_to_dict(cfg),
            run_id=wandb_id,
            resume="must" if wandb_id else None,
            job_type="training",
        )

    # Create datasets
    dset_cfg = cfg.dset if hasattr(cfg, 'dset') else cfg.get('dset', {})
    if isinstance(dset_cfg, DictConfig):
        dset_cfg = dset_cfg.to_dict()

    extra_aug = _get_cfg_value(cfg, 'extra_aug', True)
    n_extra_aug = _get_cfg_value(cfg, 'n_extra_aug', 10)
    # trainset = get_SirenAdnOriginalDataset(**dset_cfg, split="train")
    trainset = get_SirenAdnOriginalDataset(**dset_cfg, split="train")
    valset = get_SirenAdnOriginalDataset(**dset_cfg, split="val")

    if extra_aug:
        aug_dsets = []
        for i in range(1, n_extra_aug):
            aug_dsets.append(get_SirenAdnOriginalDataset(
                data_path=dset_cfg['data_path'],
                siren_path=dset_cfg['siren_path'],
                siren_prefix=f"{dset_cfg['siren_prefix']}_{i}",
                split_points=dset_cfg['split_points'],
                split="train",
            ))

        trainset = ConcatDataset([trainset] + aug_dsets)

    bs = _get_cfg_value(cfg, 'bs', 512)
    # trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    # valloader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=4)

     
    if use_ddp:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
        trainloader = DataLoader(trainset, batch_size=bs, sampler=train_sampler, num_workers=4, drop_last=True, pin_memory=True)
        # val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank, shuffle=False)
        # valloader = DataLoader(valset, batch_size=bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_sampler = None
        trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    valloader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=4)
 

    print(f"Dataset sizes: train={len(trainset)}, val={len(valset)}, batch_size={bs}.")

    sample_batch = next(iter(trainloader)) # params, init_params, img, siren_label
    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*sample_batch[0]).to("cpu"), set_all_dims=True)

    # Create model
    model_cfg = cfg.model if hasattr(cfg, 'model') else cfg.get('model', {})
    pool_cfg = cfg.pool_cls if hasattr(cfg, 'pool_cls') else cfg.get('pool_cls', {})

    if isinstance(pool_cfg, DictConfig):
        pool_cfg = pool_cfg.to_dict()

    # Create pool_cls with config
    pool_cls = partial(PerceiverPooling, **pool_cfg)
    base_model = AutoEncoder(model_cfg, spec, cfg.dataset_name, pool_cls=pool_cls).to(device)

    # Multi-GPU support with DataParallel
    # use_multi_gpu = _get_cfg_value(cfg, 'multi_gpu', False)
    # gpu_ids = _get_cfg_value(cfg, 'gpu_ids', None)  # e.g., [0, 1, 2, 3]
    # if use_multi_gpu and torch.cuda.device_count() > 1:
    #     if gpu_ids is None:
    #         gpu_ids = list(range(torch.cuda.device_count()))
    #     print(f"Using DataParallel with GPUs: {gpu_ids}")
    #     nfnet = torch.nn.DataParallel(base_model, device_ids=gpu_ids)
    # else:
    #     nfnet = base_model
    if use_ddp:
        nfnet = DDP(base_model, device_ids=[local_rank], find_unused_parameters=False)
        nfnet_fast = nfnet  # DDP에서는 compile 사용하지 않음
    else:
        use_multi_gpu = _get_cfg_value(cfg, 'multi_gpu', False)
        gpu_ids = _get_cfg_value(cfg, 'gpu_ids', None)
        if use_multi_gpu and torch.cuda.device_count() > 1:
            if gpu_ids is None:
                gpu_ids = list(range(torch.cuda.device_count()))
            nfnet = torch.nn.DataParallel(base_model, device_ids=gpu_ids)
        else:
            nfnet = base_model
        
        compile_model = _get_cfg_value(cfg, 'compile', True)
        if compile_model and not isinstance(nfnet, torch.nn.DataParallel):
            nfnet_fast = torch.compile(nfnet)
        else:
            nfnet_fast = nfnet
 
    model_for_eval = base_model  # 항상 base model로 eval
 
    watch_grads = _get_cfg_value(cfg, 'watch_grads', False)
    if watch_grads and wandb_manager.is_active:
        wandb_manager.watch(nfnet_fast, log="gradients", log_freq=1000)

    # Use unified get_batch_inr for both SIREN and FFN
    orig_batch_siren = get_batch_inr(cfg.dataset_name, inr_config=model_cfg['inr_config'])[0]
    # import pdb;pdb.set_trace()

    print(model_for_eval)
    print(f"Total params in NFN: {sum(p.numel() for p in model_for_eval.parameters()) / 1000000}M.")

    lr = _get_cfg_value(cfg, 'lr', 0.0001)
    weight_decay = _get_cfg_value(cfg, 'weight_decay', 0.01)
    opt = torch.optim.AdamW(nfnet.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = _get_cfg_value(cfg, 'total_steps', 200000)
    warmup_steps = _get_cfg_value(cfg, 'warmup_steps', 10000)
    decay_start = _get_cfg_value(cfg, 'decay_start', None)

    gradient_accumulation_steps = _get_cfg_value(cfg, 'gradient_accumulation_steps', 1)
    scheduler = get_linear_warmup_with_cos_decay(
        opt,
        total_steps=total_steps // gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        decay_start=decay_start,
    )

    # Mixed precision training - support both 'train_mixed_precision' and legacy 'amp_enabled'
    train_mixed_precision = _get_cfg_value(cfg, 'train_mixed_precision', _get_cfg_value(cfg, 'amp_enabled', False))
    amp_enabled = train_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    start_step, best_val_loss = 0, float("inf")

    # Early stopping configuration
    early_stopping_patience = _get_cfg_value(cfg, 'early_stopping_patience', None)  # None = disabled
    early_stopping_min_delta = _get_cfg_value(cfg, 'early_stopping_min_delta', 0.0)
    steps_without_improvement = 0

    if prev_ckpt is not None:
        model_for_eval.load_state_dict(prev_ckpt["nfnet"])
        opt.load_state_dict(prev_ckpt["opt"])
        start_step = prev_ckpt["step"]
        best_val_loss = prev_ckpt["best_val_loss"]
        scheduler.load_state_dict(prev_ckpt["scheduler"])
        scaler.load_state_dict(prev_ckpt["scaler"])
        steps_without_improvement = prev_ckpt.get("steps_without_improvement", 0)
        print(f"Resuming from checkpoint at step {start_step} with best_val_loss {best_val_loss:.3f} and steps_without_improvement {steps_without_improvement}.")
    else:
        print("Starting training from scratch.")
        
    batches_per_epoch = max(len(trainloader), 1)
    current_epoch = start_step // batches_per_epoch
    if hasattr(trainloader.sampler, "set_epoch"):
        trainloader.sampler.set_epoch(current_epoch)
    train_iter = iter(trainloader)
    latest_val_loss = float("nan")

    from datetime import datetime

    start_time = datetime.now()
    debug = _get_cfg_value(cfg, 'debug', False)
    debug_compile = _get_cfg_value(cfg, 'debug_compile', False)
    true_target = _get_cfg_value(cfg, 'true_target', False)
    loss_type = _get_cfg_value(cfg, 'loss_type', 'l2')
    amp_dtype_str = _get_cfg_value(cfg, 'amp_dtype', 'float16')
    grad_clip = _get_cfg_value(cfg, 'grad_clip', None)

    is_3d = cfg.data_name == 'shapenet'
    val_every = 1000 * gradient_accumulation_steps
    log_every = 100 * gradient_accumulation_steps
    # if is_3d:
    #     gradient_accumulation_steps = 8
    #     print(f"Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {bs * gradient_accumulation_steps})")

    # outer_pbar = trange(start_step, total_steps, position=0, disable=False)
    timer = SimpleTimer(total_steps)
    should_stop = torch.tensor(0, device=device)  # bool 대신 tensor

    for step in range(start_step, total_steps):
        try:
            wts_and_bs, init_wts_and_bs, img, _ = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if hasattr(trainloader.sampler, "set_epoch"):
                trainloader.sampler.set_epoch(current_epoch)
            train_iter = iter(trainloader)
            wts_and_bs, init_wts_and_bs, img, _ = next(train_iter)

        params = WeightSpaceFeatures(*wts_and_bs).to(device)
        img = orig_batch_siren(params_to_func_params(params))

        if step == start_step and debug_compile:
            target_model = nfnet_fast
            _, _, _, _, _, explanation_verbose = dynamo.explain(target_model, params)
            print(explanation_verbose)

        # Zero gradients at start of accumulation window
        if step % gradient_accumulation_steps == 0:
            opt.zero_grad()

        amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[amp_dtype_str]
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            pred_img = nfnet_fast(params)
            train_loss = compute_losses(img, pred_img)
            # Scale loss for gradient accumulation
            if gradient_accumulation_steps > 1:
                train_loss = train_loss / gradient_accumulation_steps
        scaler.scale(train_loss).backward()

        # Only update weights at end of accumulation window
        tot_norm_value = 0.0
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(opt)
            if grad_clip is not None:
                tot_norm = nn.utils.clip_grad_norm_(nfnet.parameters(), grad_clip)
                tot_norm_value = tot_norm.item()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        loss_for_log = train_loss.detach() * gradient_accumulation_steps  # Unscale for logging
        train_loss_scalar = loss_for_log.item()

        if step % val_every == 0 and is_main:
            wandb_manager.log(
                {
                    "train/loss": train_loss_scalar,
                    "lr": scheduler.get_last_lr()[0],
                    "tot_norm": tot_norm_value,
                },
                step=step,
            )

        if step % val_every == 0 and is_main:
            val_loss_scalar = float("nan")
            val_loss_tensor = evaluate(model_for_eval, valloader, orig_batch_siren, true_target, loss_type, device=device)
            val_loss_scalar = val_loss_tensor.item()
            val_metrics = {"val/loss": val_loss_scalar}
            # wandb_manager.log({"val/loss": val_loss_scalar}, step=step)
            if not is_3d:
                orig_siren, new_siren, true_img = sample_recon(model_for_eval, valloader, orig_batch_siren, device=device)
                if wandb_manager.is_active:
                    val_metrics["val/orig_siren"] = [wandb.Image(Image.fromarray(x)) for x in orig_siren]
                    val_metrics["val/recon_siren"] = [wandb.Image(Image.fromarray(x)) for x in new_siren]
                    val_metrics["val/true_img"] = [wandb.Image(Image.fromarray(x)) for x in true_img]
            wandb_manager.log(val_metrics, step=step)

            # Early stopping check
            if val_loss_scalar < best_val_loss - early_stopping_min_delta:
                torch.save(model_for_eval.state_dict(), os.path.join(cfg.inr2array_model_dir, "best.pt"))
                best_val_loss = val_loss_scalar
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            torch.save(
                {
                    "step": step,
                    "nfnet": model_for_eval.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "wandb_run_id": wandb_manager.get_run_id(),
                    "steps_without_improvement": steps_without_improvement,
                },
                ckpt_path,
            )
            latest_val_loss = val_loss_scalar
            # outer_pbar.write(f"train_loss: {train_loss_scalar:.3f}, val_loss: {latest_val_loss:.3f}")
            
            if early_stopping_patience is not None and steps_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered at step {step}. No improvement for {steps_without_improvement} validation checks.")
                wandb_manager.log({"early_stopped": True, "early_stop_step": step}, step=step)
                should_stop.fill_(1)


        if use_ddp and step % val_every == 0:
            dist.barrier()  # Ensure all processes are synced before next validation phase
            dist.broadcast(should_stop, src=0)

        if step % log_every == 0 and is_main:
            print(f"Step {step} | ETA: {timer.get_eta(step)} | train_loss: {train_loss_scalar:.3f}, val_loss: {latest_val_loss:.3f}", flush=True)

        if should_stop.item():
            break

        if debug:
            break

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Training time elapsed: {elapsed}")
    print("Starting testing")
    if is_main:
        model_for_eval.load_state_dict(torch.load(os.path.join(cfg.inr2array_model_dir, "best.pt"), map_location=device, weights_only=False))
        testset = get_SirenAdnOriginalDataset(**dset_cfg, split="test")
        testloader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        test_loss = evaluate(model_for_eval, testloader, orig_batch_siren, true_target, loss_type, device=device)

        trainset_eval = get_SirenAdnOriginalDataset(**dset_cfg, split="train")
        trainloader_eval = DataLoader(trainset_eval, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        train_loss = evaluate(model_for_eval, trainloader_eval, orig_batch_siren, true_target, loss_type, device=device)

        print(f"Train loss: {train_loss.item():.3f} | {os.path.join(cfg.inr2array_model_dir, 'best.pt')}")
        print(f"Test loss: {test_loss.item():.3f} | {os.path.join(cfg.inr2array_model_dir, 'best.pt')}")
        wandb_manager.log({"test/loss": test_loss.item(), "train/final_loss": train_loss.item()}, step=total_steps)
        wandb_manager.log_summary({
            "test/loss": test_loss.item(),
            "train/final_loss": train_loss.item(),
            "best_val_loss": best_val_loss,
        })

    # Only finish if we own the wandb manager
    if owns_wandb:
        wandb_manager.finish_run()
