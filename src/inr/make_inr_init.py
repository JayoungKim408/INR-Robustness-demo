import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import random

import torch
from torch import Tensor
from torch import nn
from tqdm import tqdm
from torch.utils._pytree import tree_map
from torch.func import functional_call, vmap, grad
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from matplotlib import pyplot as plt

from inr.siren_utils import Siren, get_mgrid

# ----------------
# Distributed init
# ----------------
def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device

# -------------
# PyTree utils
# -------------
def extract_params_buffers(model: torch.nn.Module):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return params, buffers

def stack_param_dicts(param_dicts):
    keys = param_dicts[0].keys()
    return {k: torch.stack([p[k] for p in param_dicts], dim=0) for k in keys}

def tree_apply2(f, *trees):
    return tree_map(lambda *vals: f(*vals), *trees)

# ---------------------------
# single forward / single loss
# ---------------------------
def single_forward(model, params, buffers, coords):
    return functional_call(model, (params, buffers), (coords,))[0]


def single_loss(model, params, buffers, coords, target):
    pred = single_forward(model, params, buffers, coords)
    H, W, C = target.shape
    pred = pred.reshape(H, W, C)
    return torch.nn.functional.mse_loss(pred, target)


def build_batched_loss_and_grad(model):
    g = grad(single_loss, argnums=1)
    batched_grad_fn = vmap(g, in_dims=(None, 0, None, None, 0))
    batched_loss_fn = vmap(single_loss, in_dims=(None, 0, None, None, 0))
    return batched_loss_fn, batched_grad_fn


def build_batched_data(model, params, buffers, coords_batched, shape):
    f = vmap(single_forward, in_dims=(None, 0, None, None))
    preds = f(model, params, buffers, coords_batched)
    return preds.reshape(-1, *shape)

@torch.no_grad()
def sgd_update_batched(batched_params, batched_grads, lr):
    return tree_apply2(lambda p, g: p - lr * g, batched_params, batched_grads)

# -----------------------------
# Batched Adam for per-sample params
# -----------------------------
@torch.no_grad()
def adam_init_state(batched_params):
    m = {k: torch.zeros_like(v) for k, v in batched_params.items()}
    v = {k: torch.zeros_like(v) for k, v in batched_params.items()}
    return {"m": m, "v": v, "t": 0}

@torch.no_grad()
def adam_update_batched(batched_params, batched_grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    m, v = state["m"], state["v"]

    new_m = {k: beta1 * m[k] + (1.0 - beta1) * batched_grads[k].to(batched_params[k].dtype) for k in batched_params.keys()}
    new_v = {k: beta2 * v[k] + (1.0 - beta2) * (batched_grads[k].to(batched_params[k].dtype) ** 2) for k in batched_params.keys()}

    bias_c1 = 1.0 - (beta1 ** t)
    bias_c2 = 1.0 - (beta2 ** t)
    m_hat = {k: new_m[k] / bias_c1 for k in batched_params.keys()}
    v_hat = {k: new_v[k] / bias_c2 for k in batched_params.keys()}

    new_params = {k: batched_params[k] - lr * (m_hat[k] / (v_hat[k].sqrt() + eps)) for k in batched_params.keys()}

    state["m"], state["v"] = new_m, new_v
    return new_params, state


# =========================================================
# Model creation
# =========================================================

def create_fresh_model(in_features, out_features, hidden_features, hidden_layers,
                       first_omega_0, hidden_omega_0, bias_zero_init, is_first, device):
    """Create a Siren model, optionally with zero bias init."""
    model = Siren(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        outermost_linear=True,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        is_first=is_first
    ).to(device)

    if bias_zero_init:
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)

    return model


def get_init_params(model):
    return {k: v.detach().clone() for k, v in dict(model.named_parameters()).items()}

def save_init(init_params, path):
    torch.save(init_params, path)
    print(f"  [init] saved: {path}")


# =========================================================
# Batched param preparation
# =========================================================

def prepare_batched_params_shared(base_params, B):
    """All B samples share the same init."""
    param_dicts = []
    for _ in range(B):
        copy_params = {k: v.detach().clone().requires_grad_(True) for k, v in base_params.items()}
        param_dicts.append(copy_params)
    return stack_param_dicts(param_dicts)


def prepare_batched_params_random(model_factory, B, device):
    """Each sample gets independent random init with bias=0.
    Returns (batched_params, list of init param dicts)."""
    param_dicts = []
    init_params_list = []
    for _ in range(B):
        fresh_model = model_factory()
        init_p = get_init_params(fresh_model)
        # bias=0 is already handled by model_factory (bias_zero_init=True for random_init)
        init_params_list.append(init_p)
        copy_params = {k: v.detach().clone().requires_grad_(True) for k, v in init_p.items()}
        param_dicts.append(copy_params)
    return stack_param_dicts(param_dicts), init_params_list


# =========================================================
# Flat tensor utilities for compact saving
# =========================================================

def params_dict_to_flat(params_dict):
    """Flatten a param dict {str: Tensor} into a single 1D tensor.
    Keys are sorted for deterministic ordering."""
    flat_parts = []
    param_shapes = []
    for k in sorted(params_dict.keys()):
        v = params_dict[k]
        flat_parts.append(v.detach().reshape(-1).cpu())
        param_shapes.append((k, list(v.shape)))
    return torch.cat(flat_parts, dim=0), param_shapes


def flat_to_params_dict(flat_vector, param_shapes, device='cpu'):
    """Reconstruct a param dict from a flat vector and shape info."""
    params = {}
    offset = 0
    for k, shape in param_shapes:
        numel = 1
        if isinstance(shape, Tensor):
            shape = shape.shape
        for s in shape:
            numel *= s
        params[k] = flat_vector[offset:offset+numel].reshape(shape).to(device)
        offset += numel
    return params


def save_param_shapes(param_shapes, path):
    """Save param_shapes metadata once (needed for flat <-> dict conversion)."""
    torch.save(param_shapes, path)
    print(f"  [meta] saved param_shapes: {path}")


def load_param_shapes(path):
    """Load param_shapes metadata."""
    return torch.load(path, map_location='cpu', weights_only=False)


# =========================================================
# Training
# =========================================================

def train_inr_batched_rank(
    plot_name: str,
    model_template: torch.nn.Module,
    coords_batched,
    targets_batched,
    batched_params,
    steps=5000,
    lr=5e-5,
    amp=True,
    val=False,
    save_plot=False
):
    device = targets_batched.device
    model = model_template.to(device)
    _, buffers = extract_params_buffers(model)

    batched_loss_fn, batched_grad_fn = build_batched_loss_and_grad(model)
    adam_state = adam_init_state(batched_params)

    for t in range(steps):
        if amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                grads = batched_grad_fn(model, batched_params, buffers, coords_batched, targets_batched)
        else:
            grads = batched_grad_fn(model, batched_params, buffers, coords_batched, targets_batched)

        with torch.no_grad():
            batched_params, adam_state = adam_update_batched(batched_params, grads, adam_state, lr)

            if val and (t + 1) % 1000 == 0:
                losses = batched_loss_fn(model, batched_params, buffers, coords_batched, targets_batched)
                print(f"[local step {t+1}] mean loss: {losses.mean().item():.6f}")

            if save_plot and (t + 1) % 1000 == 0:
                sample_imgs = build_batched_data(model, batched_params, buffers, coords_batched, targets_batched.shape)
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(sample_imgs[0][0].clamp(0,1).cpu().detach().numpy())
                axes[1].imshow(targets_batched[0].clamp(0,1).cpu().detach().numpy())
                plt.savefig(f"{plot_name}_step{t+1}.png")
                plt.close()

    return batched_params, buffers


# -------------------------
# MAIN
# -------------------------
DATA_MAP = {"mnist": MNIST, "fashion": FashionMNIST, "cifar": CIFAR10}

# Number of fixed inits per dataset
NUM_FIXED_INITS = {"mnist": 10, "fashion": 10, "cifar": 20}


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)

    init_mode = args.init_mode
    print(f"[init_mode] {init_mode}, [bias_zero_init] {args.bias_zero_init}")
    init_mode = init_mode + "_bz" * args.bias_zero_init + ("_no_isfirst" if not args.is_first else "")
    print(init_mode)
    DEF_TFM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])  # [-1, 1]
    print("[data] transform: ToTensor + Normalize -> [-1, 1]")

    trainset = DATA_MAP[args.dataset_name](args.data_path, transform=DEF_TFM, train=True, download=True)
    testset = DATA_MAP[args.dataset_name](args.data_path, transform=DEF_TFM, train=False, download=True)
    dataset = ConcatDataset([trainset, testset])
    if args.test:
        dataset = Subset(dataset, indices=range(0, 1000))  # Use only first 100 samples for testing
        
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    arch_dir = f"depth{args.hidden_layers}_width{args.hidden_features}_scale{args.first_omega_0}"
    # if args.bias_zero_init:
    #     arch_dir += "_bz"
    # if not args.is_first:
    #     arch_dir += "_no_isfirst"
    siren_save_path = f"{args.siren_path}/{arch_dir}/{args.dataset_name}"
    plot_path = f"plots/{arch_dir}/{args.dataset_name}/{init_mode}"
    os.makedirs(plot_path, exist_ok=True)

    # --- Init save directory ---
    init_save_dir = f"{siren_save_path}/inits/{init_mode}"
    os.makedirs(init_save_dir, exist_ok=True)

    mgrid_len = 28 if args.dataset_name in ["mnist", "fashion"] else 32
    coords_shared = get_mgrid(mgrid_len, 2).to(device)
    out_features = 3 if args.dataset_name == "cifar" else 1

    # --- Model factory ---
    def make_model():
        return create_fresh_model(
            in_features=2, out_features=out_features,
            hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
            first_omega_0=args.first_omega_0, hidden_omega_0=args.first_omega_0,
            bias_zero_init=args.bias_zero_init, is_first=args.is_first, device=device,
        )

    STEPS, LR = args.total_steps, args.lr

    # =============================================
    # Create model template
    # =============================================
    model_template = make_model()

    # --- Save param_shapes metadata once ---
    template_params = get_init_params(model_template)
    _, param_shapes = params_dict_to_flat(template_params)
    save_param_shapes(param_shapes, os.path.join(siren_save_path, "param_shapes.pth"))
    flat_dim = sum(v.numel() for v in template_params.values())
    print(f"[flat save] flat_dim={flat_dim}, keys={[k for k,_ in param_shapes]}")

    # =============================================
    # FIXED_INIT mode
    # =============================================
    if init_mode.startswith("fixed_init"):
        num_inits = NUM_FIXED_INITS[args.dataset_name]
        print(f"[fixed_init] {num_inits} inits x {len(dataset)} samples = {num_inits * len(dataset)} total INRs")
        # prefix = "fixed_init" + "_bz" * args.bias_zero_init + ("_no_isfirst" if not args.is_first else "")

        # --- Create and save init pool ---
        init_pool = []
        for pool_idx in range(num_inits):
            if pool_idx == 0:
                init_p = get_init_params(model_template)
            else:
                fresh = make_model()
                init_p = get_init_params(fresh)
            init_pool.append(init_p)
            save_init(init_p, os.path.join(init_save_dir, f"init_{pool_idx}.pth"))
        print(f"[fixed_init] saved {num_inits} init files")

        # --- Create output directories ---
        for init_id in range(num_inits):
            for label in range(10):
                dir_name = f"{init_mode}_{init_id}" if init_id != 0 else init_mode
                dir_path = f"{siren_save_path}/{dir_name}/{label}s"
                os.makedirs(dir_path, exist_ok=True)

        def return_path(init_id, label, global_idx):
            dir_name = f"{init_mode}_{init_id}" if init_id != 0 else init_mode
            return f"{siren_save_path}/{dir_name}/{label}s/net{global_idx}.pth"

        # --- For each init, train on entire dataset ---
        for init_id in range(num_inits):
            print(f"\n{'='*60}")
            print(f"[fixed_init] Pass {init_id+1}/{num_inits} (init_{init_id})")
            print(f"{'='*60}")

            current_init = init_pool[init_id]
            init_flat, _ = params_dict_to_flat(current_init)

            total_loss = 0
            for b_idx, (targets_b, label) in tqdm(enumerate(dataloader), total=len(dataloader),
                                                    desc=f"init_{init_id}"):
                targets_b = torch.permute(targets_b, (0, 2, 3, 1)).to(device, non_blocking=True)
                B = targets_b.shape[0]

                batched_params = prepare_batched_params_shared(current_init, B)

                final_params_local, buffers_local = train_inr_batched_rank(
                    f"{plot_path}/init{init_id}_batch{b_idx}",
                    model_template,
                    coords_batched=coords_shared,
                    targets_batched=targets_b,
                    batched_params=batched_params,
                    steps=STEPS, lr=LR, amp=True, val=True,
                    save_plot=(init_id == 0 and b_idx % 100 == 0)
                )

                blf, _ = build_batched_loss_and_grad(model_template)
                with torch.no_grad():
                    loss_local = blf(model_template, final_params_local, buffers_local,
                                     coords_shared, targets_b).mean()
                total_loss += loss_local.item()

                if (b_idx + 1) % 50 == 0:
                    print(f"  [init_{init_id}] batch {b_idx+1} mean loss: {loss_local.item():.6f}")

                for i in range(B):
                    global_idx = b_idx * args.bs + i
                    sample_label = label[i].item()
                    save_path = return_path(init_id, sample_label, global_idx)

                    sample_params = {k: v[i] for k, v in final_params_local.items()}
                    params_flat, _ = params_dict_to_flat(sample_params)

                    torch.save({"params": params_flat, "init": init_flat}, save_path)

            avg_loss = total_loss / len(dataloader)
            print(f"[fixed_init] init_{init_id} done, avg loss: {avg_loss:.6f}")

    # =============================================
    # RANDOM_INIT mode
    # =============================================
    elif init_mode.startswith("random_init"):
        print(f"[random_init] 1 copy, independent random init per sample (bias=0)")
        # prefix = "random_init" + "_bz" * args.bias_zero_init + ("_no_isfirst" if not args.is_first else "")

        for label in range(10):
            dir_path = f"{siren_save_path}/{init_mode}/{label}s"
            os.makedirs(dir_path, exist_ok=True)

        def return_path(label, global_idx):
            return f"{siren_save_path}/{init_mode}/{label}s/net{global_idx}.pth"

        total_loss = 0
        for b_idx, (targets_b, label) in tqdm(enumerate(dataloader), total=len(dataloader),
                                                desc="random_init"):
            targets_b = torch.permute(targets_b, (0, 2, 3, 1)).to(device, non_blocking=True)
            B = targets_b.shape[0]

            batched_params, per_sample_inits = prepare_batched_params_random(make_model, B, device)

            final_params_local, buffers_local = train_inr_batched_rank(
                f"{plot_path}/random_batch{b_idx}",
                model_template,
                coords_batched=coords_shared,
                targets_batched=targets_b,
                batched_params=batched_params,
                steps=STEPS, lr=LR, amp=True, val=True,
                save_plot=b_idx % 100 == 0
            )

            blf, _ = build_batched_loss_and_grad(model_template)
            with torch.no_grad():
                loss_local = blf(model_template, final_params_local, buffers_local,
                                 coords_shared, targets_b).mean()
            total_loss += loss_local.item()

            if (b_idx + 1) % 50 == 0:
                print(f"  batch {b_idx+1} mean loss: {loss_local.item():.6f}")

            for i in range(B):
                global_idx = b_idx * args.bs + i
                sample_label = label[i].item()
                save_path = return_path(sample_label, global_idx)

                sample_params = {k: v[i] for k, v in final_params_local.items()}
                params_flat, _ = params_dict_to_flat(sample_params)
                init_flat, _ = params_dict_to_flat(per_sample_inits[i])

                torch.save({"params": params_flat, "init": init_flat}, save_path)

        avg_loss = total_loss / len(dataloader)
        print(f"[random_init] done, avg loss: {avg_loss:.6f}")

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}. Use 'fixed_init' or 'random_init'.")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='mnist')
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--data_path", type=str, default="/mnt/data0/jayoung/INR-Robustness/assets/raw")
    parser.add_argument("--siren_path", type=str, default="/mnt/data0/jayoung/INR-Robustness/assets/sirens/test")

    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--first_omega_0", type=int, default=30)
    parser.add_argument("--hidden_omega_0", type=float, default=30.0)

    # --- Model options ---
    parser.add_argument("--bias_zero_init", type=str2bool, default=False, # True for DWS (random_init), False for others
                        help="Zero-initialize all bias parameters")
    parser.add_argument("--is_first", type=str2bool, default=True, # False for DWS (random_init), True for others
                        help="Whether to use is_first=True in the first layer of the SIREN")

    # --- Init mode ---
    parser.add_argument("--init_mode", type=str, default="fixed_init",
                        choices=["fixed_init", "random_init"],
                        help="fixed_init = N fixed inits x full dataset, "
                             "random_init = independent random init per sample (bias=0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", type=str2bool, default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)