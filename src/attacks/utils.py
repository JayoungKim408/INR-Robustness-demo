'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.func import functional_call, grad, vmap
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from datetime import timedelta

from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures, params_to_state_dicts
from basemodels.utils.nfn.siren_utils import get_mgrid, Siren
from basemodels.utils.nfn.data_utils import state_dict_to_tensors
from basemodels.utils.nfn.train_utils import set_seed

# Default eps_std for embedding-space PGD attack (std-normalized epsilon)
# These values are calibrated to match image-space attack strength
DEFAULT_EPS = {"cifar": 0.03, "mnist": 0.3, "fashion": 0.1, "shapenet": 0.025}
DEFAULT_EPS_STD = {"cifar": 0.12, "mnist": 0.97, "fashion": 0.28, "shapenet": 0.0}
SPLIT_POINTS = {"cifar": [45000, 50000], "mnist": [55000, 60000], "fashion": [55000, 60000], "shapenet": None}  # Train/Val split at 55k for MNIST and FashionMNIST

class SimpleTimer:
    def __init__(self, total_steps, start_step=0):
        self.total = total_steps
        self.start_step = start_step
        self.start = time.time()

    def get_eta(self, current_step):
        elapsed = time.time() - self.start
        steps_done = current_step - self.start_step
        speed = steps_done / elapsed if elapsed > 0 and steps_done > 0 else 0
        eta = (self.total - current_step) / speed if speed > 0 else 0
        return str(timedelta(seconds=int(eta)))


def get_dataset_params(dataset_name: str) -> dict:
    """Return common dataset-specific parameters.

    Args:
        dataset_name: One of 'mnist', 'fashion', 'cifar'.

    Returns:
        Dict with keys 'sidelen', 'c_out', 'num_classes'.
    """
    if dataset_name in ("mnist", "fashion"):
        return {"sidelen": 28, "c_out": 1, "num_classes": 10}
    elif dataset_name == "cifar":
        return {"sidelen": 32, "c_out": 3, "num_classes": 10}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def wsfeat_to_dws(wsfeat: WeightSpaceFeatures):
    """
    Convert WeightSpaceFeatures to DWS format.

    WeightSpaceFeatures stores weights as (batch, 1, out_features, in_features)
    DWS expects weights as (batch, in_features, out_features, 1) - transposed + feature dim

    WeightSpaceFeatures stores biases as (batch, 1, out_features)
    DWS expects biases as (batch, out_features, 1) - with feature dim

    Args:
        wsfeat: WeightSpaceFeatures object

    Returns:
        Tuple of (weights, biases) in DWS format
    """
    # Weights: (batch, 1, out, in) -> squeeze -> (batch, out, in) -> transpose -> (batch, in, out) -> unsqueeze -> (batch, in, out, 1)
    weights = tuple(
        w.squeeze(1).transpose(1, 2).unsqueeze(-1) for w in wsfeat.weights
    )
    # Biases: (batch, 1, out) -> squeeze -> (batch, out) -> unsqueeze -> (batch, out, 1)
    biases = tuple(
        b.squeeze(1).unsqueeze(-1) for b in wsfeat.biases
    )
    return weights, biases


def dws_to_wsfeat(weights, biases):
    """
    Convert DWS format back to WeightSpaceFeatures.

    DWS weights: (batch, in_features, out_features, 1)
    WeightSpaceFeatures weights: (batch, 1, out_features, in_features)

    DWS biases: (batch, out_features, 1)
    WeightSpaceFeatures biases: (batch, 1, out_features)

    Args:
        weights: Tuple of weight tensors in DWS format
        biases: Tuple of bias tensors in DWS format

    Returns:
        WeightSpaceFeatures object
    """
    # Weights: (batch, in, out, 1) -> squeeze -> (batch, in, out) -> transpose -> (batch, out, in) -> unsqueeze -> (batch, 1, out, in)
    wsfeat_weights = tuple(
        w.squeeze(-1).transpose(1, 2).unsqueeze(1) for w in weights
    )
    # Biases: (batch, out, 1) -> squeeze -> (batch, out) -> unsqueeze -> (batch, 1, out)
    wsfeat_biases = tuple(
        b.squeeze(-1).unsqueeze(1) for b in biases
    )
    return WeightSpaceFeatures(wsfeat_weights, wsfeat_biases)


class BatchedINRBuilder:
    def __init__(self,
                 sidelen,
                 hidden_features=32,
                 hidden_layers=1,
                 first_omega_0=30.,
                 hidden_omega_0=30.,
                 c_out=1,
                 steps=5000,
                 lr=5e-4,
                 amp=True,
                 device=None,
                 verbose=False
                 ):

        set_seed(42)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.steps = steps
        self.c_out = c_out
        self.lr = lr
        self.amp = amp
        self.verbose = verbose
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sidelen = sidelen

        self._cached_model_template = None
        self._cached_grad_fn = None
        self._cached_loss_fn = None
    
    def _get_batched_fns(self):
        if self._cached_grad_fn is None:
            self._cached_model_template = self._create_model_template()
            self._cached_loss_fn, self._cached_grad_fn = \
                self._build_batched_loss_and_grad(self._cached_model_template)
        return (self._cached_model_template, 
                self._cached_loss_fn, 
                self._cached_grad_fn)


    def _create_model_template(self):
        return Siren(
            in_features=2,
            out_features=self.c_out,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            outermost_linear=True,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0
        ).to(self.device)

    @staticmethod
    def _extract_params_buffers(model: nn.Module):
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        return params, buffers

    @staticmethod
    def _stack_param_dicts(param_dicts):
        keys = param_dicts[0].keys()
        return {k: torch.stack([p[k] for p in param_dicts], dim=0) for k in keys}

    @staticmethod
    def _single_forward(model, params, buffers, coords):
        return functional_call(model, (params, buffers), (coords,))[0]

    @staticmethod
    def _single_loss(model, params, buffers, coords, target):
        pred = BatchedINRBuilder._single_forward(model, params, buffers, coords)
        H, W, C = target.shape
        pred = pred.reshape(H, W, C)
        return torch.nn.functional.mse_loss(pred, target)

    @staticmethod
    def _build_batched_loss_and_grad(model):
        g = grad(BatchedINRBuilder._single_loss, argnums=1)
        batched_grad_fn = vmap(g, in_dims=(None, 0, None, None, 0))
        batched_loss_fn = vmap(BatchedINRBuilder._single_loss, in_dims=(None, 0, None, None, 0))
        return batched_loss_fn, batched_grad_fn

    @staticmethod
    @torch.no_grad()
    def _adam_init_state(batched_params):
        m = {k: torch.zeros_like(v) for k, v in batched_params.items()}
        v = {k: torch.zeros_like(v) for k, v in batched_params.items()}
        return {"m": m, "v": v, "t": 0}

    @staticmethod
    @torch.no_grad()
    def _adam_update_batched(batched_params, batched_grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
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

        
    def _load_wts_and_bs(self, wsfeat):
        """Load wsfeat directly into batched_params without per-sample loop."""
        if self._cached_model_template is None:
            self._cached_model_template = self._create_model_template()
        model_template = self._cached_model_template
        keys = list(dict(model_template.named_parameters()).keys())

        batched_params = {}
        layer_idx = 0
        for i in range(len(wsfeat.weights)):
            batched_params[keys[layer_idx]] = (
                wsfeat.weights[i].squeeze(1).to(self.device).clone().requires_grad_(True)
            )
            layer_idx += 1
            batched_params[keys[layer_idx]] = (
                wsfeat.biases[i].squeeze(1).to(self.device).clone().requires_grad_(True)
            )
            layer_idx += 1

        return model_template, batched_params, {}

    def build_wts_and_bs_from_img(self, img_tensor, init_params=None):
        import time
        device = self.device
        batch_size, c_in, h, w = img_tensor.shape
        sidelen = h

        t0 = time.time()
        targets_batched = img_tensor.permute(0, 2, 3, 1).contiguous().to(device)
        
        # coords와 grad_fn 캐싱
        if not hasattr(self, '_cached_coords') or self._cached_coords.device != torch.device(device):
            self._cached_coords = get_mgrid(sidelen, 2).to(device)
        coords_shared = self._cached_coords
        
        if self._cached_model_template is None:
            self._cached_model_template = self._create_model_template()
        model_template = self._cached_model_template
        
        if not hasattr(self, '_cached_batched_grad_fn'):
            _, self._cached_batched_grad_fn = self._build_batched_loss_and_grad(model_template)
        batched_grad_fn = self._cached_batched_grad_fn
        torch.cuda.synchronize()
        # t1 = time.time()

        batched_params = self._load_params_only(init_params)
        torch.cuda.synchronize()
        # t2 = time.time()

        adam_state = self._adam_init_state(batched_params)
        
        for step in range(self.steps):
            if self.amp and self.device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    grads = batched_grad_fn(model_template, batched_params,
                                        {}, coords_shared, targets_batched)
            else:
                grads = batched_grad_fn(model_template, batched_params,
                                    {}, coords_shared, targets_batched)
            with torch.no_grad():
                batched_params, adam_state = self._adam_update_batched(
                    batched_params, grads, adam_state, self.lr)
        torch.cuda.synchronize()
        # t3 = time.time()

        wts_and_bs = self._batched_params_to_wts_and_bs(batched_params)

        return wts_and_bs


    def _load_params_only(self, wsfeat):
        """Load wsfeat into batched_params dict. No model_template creation."""
        keys = list(dict(self._cached_model_template.named_parameters()).keys())
        batched_params = {}
        layer_idx = 0
        for i in range(len(wsfeat.weights)):
            batched_params[keys[layer_idx]] = (
                wsfeat.weights[i].squeeze(1).to(self.device).clone().requires_grad_(True)
            )
            layer_idx += 1
            batched_params[keys[layer_idx]] = (
                wsfeat.biases[i].squeeze(1).to(self.device).clone().requires_grad_(True)
            )
            layer_idx += 1
        return batched_params


    @staticmethod
    def _batched_params_to_wts_and_bs(batched_params):
        """Convert batched_params dict directly to (weights, biases) tuples."""
        weights = []
        biases = []
        keys = sorted(batched_params.keys())
        for k in keys:
            if k.endswith("weight"):
                weights.append(batched_params[k].detach().unsqueeze(1))
            elif k.endswith("bias"):
                biases.append(batched_params[k].detach().unsqueeze(1))
        return (tuple(weights), tuple(biases))

    def build_img_from_wts_and_bs(self, wts_and_bs, if_wsfeat=False, requires_grad=False):
        if if_wsfeat:
            wsfeat = wts_and_bs
        else:
            wsfeat = WeightSpaceFeatures(*wts_and_bs).to(self.device)
        coords_shared = get_mgrid(self.sidelen, 2).to(self.device)
        model_template = self._create_model_template()
        keys = list(dict(model_template.named_parameters()).keys())
        
        state_dicts = params_to_state_dicts(keys, wsfeat)
        stacked = {}
        for key in keys:
            stacked[key] = torch.stack([sd[key] for sd in state_dicts])

        def single_call(params):
            out, _ = functional_call(model_template, params, (coords_shared,))
            return out

        imgs = vmap(single_call)(stacked)
        imgs = imgs.view(-1, self.sidelen, self.sidelen, self.c_out)
        imgs = (imgs + 1) / 2
        
        if not requires_grad:
            imgs = imgs.detach()
        
        return imgs
    
    def build_wts_and_bs_from_pcd(self, pcds, init_params=None,
                               num_queries_on_surface=10000,
                               stds=[0.003, 0.01, 0.1],
                               num_points_per_std=[5000, 4000, 500, 500],
                               num_points_fitting=10000):
        from basemodels.inr2vec.pcd_retrieval_and_classification.pcd import compute_udf_from_pcd
        """
        pcds: [B, N, 3] point clouds in [-1, 1]
        Returns batched wts_and_bs (same format as build_wts_and_bs_from_img)
        """
        device = self.device
        batch_size = pcds.shape[0]
        pcds = pcds.to(device)

        # Generate UDF query points for each PCD
        coords_list, labels_list = [], []
        for j in range(batch_size):
            c, l = compute_udf_from_pcd(
                pcds[j], num_queries_on_surface, stds, num_points_per_std,
                coords_range=(-1, 1), convert_to_bce_labels=True,
            )
            coords_list.append(c)
            labels_list.append(l)
        
        all_coords = torch.stack(coords_list, dim=0).to(device)    # (B, M, 3)
        all_labels = torch.stack(labels_list, dim=0).to(device)     # (B, M)

        # Build model template (3D SIREN)
        model_template = Siren(
            in_features=3, out_features=1,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            outermost_linear=True,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0,
        ).to(device)

        keys = list(dict(model_template.named_parameters()).keys())

        if init_params is not None:
            base_params, buffers = params_to_state_dicts(keys, init_params), {}
            param_dicts = [{k: v.to(device).clone().requires_grad_(True) for k, v in p.items()} for p in base_params]
            batched_params = self._stack_param_dicts(param_dicts)
        else:
            single_params, buffers = self._extract_params_buffers(model_template)
            batched_params = {k: v.unsqueeze(0).expand(batch_size, *v.shape).clone().requires_grad_(True)
                            for k, v in single_params.items()}

        # BCE loss for UDF
        def single_bce_loss(model, params, buffers, coords, labels):
            pred = functional_call(model, (params, buffers), (coords,))[0].squeeze(-1)  # (M,)
            return F.binary_cross_entropy_with_logits(pred, labels)

        batched_grad_fn = vmap(grad(single_bce_loss, argnums=1), in_dims=(None, 0, None, 0, 0))
        adam_state = self._adam_init_state(batched_params)

        for step in range(self.steps):
            # Random subsample query points per step
            if num_points_fitting < all_coords.shape[1]:
                idx = torch.randint(0, all_coords.shape[1], (num_points_fitting,), device=device)
                coords_batch = all_coords[:, idx]
                labels_batch = all_labels[:, idx]
            else:
                coords_batch = all_coords
                labels_batch = all_labels

            grads = batched_grad_fn(model_template, batched_params, buffers, coords_batch, labels_batch)
            with torch.no_grad():
                batched_params, adam_state = self._adam_update_batched(batched_params, grads, adam_state, self.lr)

        # Convert to wts_and_bs format
        state_dicts = [{k: v[i].detach().clone() for k, v in batched_params.items()} for i in range(batch_size)]
        wts_and_bs = []
        for sd in state_dicts:
            weights, biases = state_dict_to_tensors(sd)
            wts_and_bs.append((weights, biases))
        wts_and_bs = torch.utils.data.default_collate(wts_and_bs)
        return wts_and_bs, state_dicts

    @torch.no_grad()
    def visualize_reconstruction(self, wts_and_bs, x, wts_and_bs_adv=None, x_adv=None, eps=None, steps=None, norm=None, plot_dir=".", object="Perturbed", title="Reconstruction"):
        x = torch.permute(x, (0, 2, 3, 1)).to(self.device, non_blocking=True)
        coords_shared = get_mgrid(self.sidelen, 2).to(self.device)

        def recon(wts):
            mt, bp, _ = self._load_wts_and_bs(wts)
            out = vmap(self._single_forward, in_dims=(None, 0, None, None))(mt, bp, {}, coords_shared)
            return out.view(-1, self.sidelen, self.sidelen, self.c_out)

        def to_img(t):
            return ((t.detach().cpu() + 1.0) / 2.0).clamp(0, 1).numpy()

        recon_clean = recon(wts_and_bs)
        ref = x  # MSE reference

        panels = [(x, "Original", None), (recon_clean, "Recon", x)]
        if x_adv is not None:
            x_adv = torch.permute(x_adv, (0, 2, 3, 1)).to(self.device, non_blocking=True)
            panels.append((x_adv, object, None))
            ref = x_adv
        if wts_and_bs_adv is not None:
            panels.append((recon(wts_and_bs_adv), f"{object} Recon", ref))

        num_show = min(4, x.shape[0])
        fig, axes = plt.subplots(num_show, len(panels), figsize=(2 * len(panels), 2 * num_show))
        if eps:
            fig.suptitle(f"eps={eps}, steps={steps}, norm={norm}", y=0.98)
        if num_show == 1:
            axes = np.array([axes])
        for i in range(num_show):
            for j, (data, label, mse_ref) in enumerate(panels):
                axes[i, j].imshow(to_img(data[i]))
                if mse_ref is not None:
                    mse = torch.mean((mse_ref[i] - data[i]) ** 2).item()
                    axes[i, j].set_title(f"{label}\n(MSE={mse:.4f})")
                else:
                    axes[i, j].set_title(label)
                axes[i, j].axis("off")

        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{title}.png")
        plt.close(fig)


def infer_eps_linf(x, x_adv, input_scale='-1,1'):
    if isinstance(x, torch.Tensor):
        X = x.clone().detach().float()
        XA = x_adv.clone().detach().float()
    else:
        X = torch.tensor(x).float()
        XA = torch.tensor(x_adv).float()

    if X.dim() == 3:  # (C,H,W)
        X = X.unsqueeze(0); XA = XA.unsqueeze(0)
    elif X.dim() == 2:  # (H,W)
        X = X.unsqueeze(0).unsqueeze(0); XA = XA.unsqueeze(0).unsqueeze(0)

    if input_scale == '-1,1':
        X = (X + 1) / 2.0
        XA = (XA + 1) / 2.0
    elif input_scale != '0-1':
        raise ValueError("input_scale must be '0-1' or '-1,1'.")

    delta = (XA - X).abs().view(X.size(0), -1)   # (B, d)
    eps_linf_per_sample = delta.max(dim=1)[0]    # (B,)
    eps_l2_per_sample = delta.norm(p=2, dim=1)   # (B,)

    eps_linf_max = eps_linf_per_sample.max().item()
    eps_linf_mean = eps_linf_per_sample.mean().item()

    return {
        'eps_linf_max_0_1': eps_linf_max,
        'eps_linf_mean_0_1': eps_linf_mean,
        'eps_linf_max_pixels': eps_linf_max * 255.0,
        'eps_l2_max': eps_l2_per_sample.max().item(),
        'eps_l2_mean': eps_l2_per_sample.mean().item(),
        'eps_l2_max_pixels': eps_l2_per_sample.max().item() * 255.0,
    }





def compute_perturbation_stats(x_nat, x_adv):
    assert x_nat.shape == x_adv.shape
    device = x_nat.device

    # delta
    delta = (x_adv - x_nat).detach().contiguous()

    # per-sample Linf, L2, and L1
    B = delta.shape[0]
    delta_flat = delta.view(B, -1)
    linf = delta_flat.abs().max(dim=1)[0]            # (B,)
    l2 = torch.norm(delta_flat, p=2, dim=1)          # (B,)
    l1 = torch.norm(delta_flat, p=1, dim=1)          # (B,)

    # basic stats
    stats = {
        'linf_max': float(linf.max().cpu().item()),
        'linf_mean': float(linf.mean().cpu().item()),
        'linf_median': float(linf.median().cpu().item()),
        'linf_std': float(linf.std().cpu().item()),
        'l2_max': float(l2.max().cpu().item()),
        'l2_mean': float(l2.mean().cpu().item()),
        'l2_median': float(l2.median().cpu().item()),
        'l2_std': float(l2.std().cpu().item()),
        'l1_max': float(l1.max().cpu().item()),
        'l1_mean': float(l1.mean().cpu().item()),
        'l1_median': float(l1.median().cpu().item()),
        'l1_std': float(l1.std().cpu().item()),
        'num_samples': int(B),
    }

    # return raw arrays too (cpu numpy) for hist/plots if needed
    arrays = {
        'linf': linf,
        'l2': l2,
        'l1': l1,
        'delta': delta,
    }

    return stats, arrays


def assert_within_eps(x_nat, x_adv, eps, norm='linf', tol=1e-6, raise_err=False):
    stats, arrays = compute_perturbation_stats(x_nat, x_adv)
    linf_arr = arrays['linf']
    l2_arr = arrays['l2']
    l1_arr = arrays['l1']

    if norm == 'linf':
        violations = (linf_arr > (eps + tol))
        num_viol = int(violations.sum())
        ok = (num_viol == 0)
        report = {
            'ok': ok,
            'num_violations': num_viol,
            'violation_ratio': float(num_viol / stats['num_samples']),
            'max_linf': stats['linf_max'],
            'eps': eps
        }
    elif norm == 'l2':
        violations = (l2_arr > (eps + tol))
        num_viol = int(violations.sum())
        ok = (num_viol == 0)
        report = {
            'ok': ok,
            'num_violations': num_viol,
            'violation_ratio': float(num_viol / stats['num_samples']),
            'max_l2': stats['l2_max'],
            'max_linf': stats['linf_max'],
            'eps': eps
        }
    elif norm == 'l1':
        violations = (l1_arr > (eps + tol))
        num_viol = int(violations.sum())
        ok = (num_viol == 0)
        report = {
            'ok': ok,
            'num_violations': num_viol,
            'violation_ratio': float(num_viol / stats['num_samples']),
            'max_l1': stats['l1_max'],
            'max_linf': stats['linf_max'],
            'eps': eps
        }
    else:
        raise ValueError("norm must be 'linf', 'l2', or 'l1'")

    if raise_err and not report['ok']:
        raise AssertionError(f"Perturbation exceeds {norm} eps: {report}")

    return report, stats, arrays


DATA_MAP = {
    "cifar": torchvision.datasets.CIFAR10,
    "fashion": torchvision.datasets.FashionMNIST,
    "mnist": torchvision.datasets.MNIST,
}

def load_data(dname, data_path=None):
    DEF_TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    
    trainset = DATA_MAP[dname](root=data_path, train=True, download=True, transform=DEF_TFM)
    testset = DATA_MAP[dname](root=data_path, train=False, download=True, transform=DEF_TFM)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    return trainloader, testloader