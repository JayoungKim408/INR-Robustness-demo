"""
Quick verification of saved INR files.

Usage:
    python src/inr/verify_inrs.py --dataset mnist
    python src/inr/verify_inrs.py --dataset cifar --init_mode init_pool
    python src/inr/verify_inrs.py  # checks all datasets × all modes
"""

import os
import sys
import argparse
import random

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import ConcatDataset

sys.path.insert(0, os.path.dirname(__file__))
from siren_utils import Siren, get_mgrid

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_LAYERS   = 1
HIDDEN_FEATURES = 32
FIRST_OMEGA_0   = 30

EXPECTED_COUNT = {"mnist": 70_000, "fashion": 70_000, "cifar": 60_000}
POOL_SIZE      = {"mnist": 10,     "fashion": 10,      "cifar": 20}
N_SPOT_CHECKS  = 5   # files to load and forward-pass

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_siren(out_features, device):
    return Siren(
        in_features=2,
        out_features=out_features,
        hidden_features=HIDDEN_FEATURES,
        hidden_layers=HIDDEN_LAYERS,
        outermost_linear=True,
        first_omega_0=FIRST_OMEGA_0,
        hidden_omega_0=FIRST_OMEGA_0,
    ).to(device)


def load_pth(path, device):
    return torch.load(path, map_location=device, weights_only=True)


# ── Checks ────────────────────────────────────────────────────────────────────
def check_file_count(siren_root, dataset, init_mode):
    """Count net*.pth files across all label subdirs."""
    total = 0
    missing_dirs = []
    for label in range(10):
        d = os.path.join(siren_root, f"{init_mode}_{label}s")
        if not os.path.isdir(d):
            missing_dirs.append(d)
            continue
        count = len([f for f in os.listdir(d) if f.startswith("net") and f.endswith(".pth")])
        total += count
    expected = EXPECTED_COUNT[dataset]
    ok = total == expected and not missing_dirs
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] file count: {total} / {expected}")
    if missing_dirs:
        for d in missing_dirs:
            print(f"         missing dir: {d}")
    return ok


def check_init_files(siren_root, dataset, init_mode):
    """Verify init .pth files exist (shared_init and init_pool)."""
    ok = True
    if init_mode == "shared_init":
        path = os.path.join(siren_root, "inits", "shared_init", "shared_init.pth")
        exists = os.path.isfile(path)
        print(f"  [{'OK' if exists else 'FAIL'}] shared init file: {path}")
        ok = exists
    elif init_mode == "init_pool":
        pool_dir = os.path.join(siren_root, "inits", "init_pool")
        n = POOL_SIZE[dataset]
        found = 0
        for i in range(n):
            p = os.path.join(pool_dir, f"pool_id{i}.pth")
            if os.path.isfile(p):
                found += 1
            else:
                print(f"         missing pool file: {p}")
        print(f"  [{'OK' if found == n else 'FAIL'}] pool files: {found} / {n}")
        ok = found == n
    return ok


def spot_check(siren_root, dataset, init_mode, device, n=N_SPOT_CHECKS):
    """Load N random files, verify structure, run forward pass, compute MSE."""
    out_features = 3 if dataset == "cifar" else 1
    mgrid_len = 32 if dataset == "cifar" else 28
    coords = get_mgrid(mgrid_len, 2).to(device)

    model = make_siren(out_features, device)

    all_paths = []
    for label in range(10):
        d = os.path.join(siren_root, f"{init_mode}_{label}s")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("net") and f.endswith(".pth"):
                all_paths.append(os.path.join(d, f))

    if not all_paths:
        print("  [FAIL] no files found for spot check")
        return False

    sample = random.sample(all_paths, min(n, len(all_paths)))
    errors = []
    losses = []

    for path in sample:
        try:
            d = load_pth(path, device)
            # Check keys
            assert "params" in d and "init" in d, f"missing keys: {list(d.keys())}"
            params = d["params"]
            # Forward pass with saved params
            with torch.no_grad():
                out = model.forward_with_params(params, coords) if hasattr(model, "forward_with_params") else _functional_forward(model, params, coords)
            # MSE against zero (just a shape/crash check — real MSE needs the image)
            assert out.shape[-1] == out_features, f"unexpected output shape: {out.shape}"
            losses.append(out.abs().mean().item())  # proxy: just mean activation
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False
    print(f"  [OK]   spot-checked {len(sample)} files — no load/shape errors")
    return True


def _functional_forward(model, params, coords):
    """Functional forward using saved param dict (no vmap)."""
    from torch.func import functional_call
    params_with_grad = {k: v.detach() for k, v in params.items()}
    return functional_call(model, params_with_grad, (coords,))[0]


def check_init_consistency(siren_root, dataset, init_mode, device, n=20):
    """
    shared_init: all sampled files must have identical init tensors.
    random_init: sampled files must have diverse init tensors.
    init_pool:   init tensors should match one of the pool members.
    """
    all_paths = []
    for label in range(100):
        d = os.path.join(siren_root, f"{init_mode}_{label}s")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("net") and f.endswith(".pth"):
                all_paths.append(os.path.join(d, f))

    if not all_paths:
        return True

    sample = random.sample(all_paths, min(n, len(all_paths)))
    inits = [load_pth(p, device)["init"] for p in sample]

    # Pick first key for comparison
    key = next(iter(inits[0].keys()))

    if init_mode == "shared_init":
        ref = inits[0][key]
        all_same = all(torch.allclose(d[key], ref) for d in inits[1:])
        print(f"  [{'OK' if all_same else 'FAIL'}] shared_init: all sampled inits identical: {all_same}")
        return all_same

    elif init_mode == "random_init":
        ref = inits[0][key]
        any_diff = any(not torch.allclose(d[key], ref) for d in inits[1:])
        print(f"  [{'OK' if any_diff else 'FAIL'}] random_init: sampled inits are diverse: {any_diff}")
        return any_diff

    elif init_mode == "init_pool":
        pool_dir = os.path.join(siren_root, "inits", "init_pool")
        pool_inits = []
        i = 0
        while True:
            p = os.path.join(pool_dir, f"pool_id{i}.pth")
            if not os.path.isfile(p):
                break
            pool_inits.append(load_pth(p, device))
            i += 1
        if not pool_inits:
            print("  [SKIP] init_pool: no pool files found for consistency check")
            return True
        pool_key_tensors = [d[key] for d in pool_inits]
        mismatches = 0
        for d in inits:
            t = d[key]
            if not any(torch.allclose(t, pt) for pt in pool_key_tensors):
                mismatches += 1
        print(f"  [{'OK' if mismatches == 0 else 'FAIL'}] init_pool: {mismatches}/{len(inits)} inits not matching any pool member")
        return mismatches == 0

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def verify(siren_path, dataset, init_mode, device):
    siren_root = os.path.join(
        siren_path,
        f"depth{HIDDEN_LAYERS}_width{HIDDEN_FEATURES}_scale{FIRST_OMEGA_0}",
        dataset,
    )
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  |  Mode: {init_mode}")
    print(f"  Root: {siren_root}")
    print(f"{'='*60}")

    if not os.path.isdir(siren_root):
        print(f"  [FAIL] siren_root does not exist: {siren_root}")
        return False

    results = []
    results.append(check_file_count(siren_root, dataset, init_mode))
    results.append(check_init_files(siren_root, dataset, init_mode))
    results.append(spot_check(siren_root, dataset, init_mode, device))
    results.append(check_init_consistency(siren_root, dataset, init_mode, device))

    overall = all(results)
    print(f"  → {'PASS' if overall else 'FAIL'}")
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siren_path", default="/nfs/stak/users/hongsa/INR-Robustness/assets/sirens")
    parser.add_argument("--dataset", choices=["mnist", "fashion", "cifar", "all"], default="all")
    parser.add_argument("--init_mode", choices=["shared_init", "init_pool", "random_init", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    datasets   = ["mnist", "fashion", "cifar"] if args.dataset   == "all" else [args.dataset]
    init_modes = ["shared_init", "init_pool", "random_init"] if args.init_mode == "all" else [args.init_mode]

    results = []
    for ds in datasets:
        for mode in init_modes:
            results.append(verify(args.siren_path, ds, mode, device))

    print(f"\n{'='*60}")
    n_pass = sum(results)
    n_total = len(results)
    print(f"  Overall: {n_pass}/{n_total} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
