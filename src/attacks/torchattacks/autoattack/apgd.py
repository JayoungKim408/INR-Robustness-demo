"""
Auto-PGD (APGD) adapted for INR weight space with ICOP projection.
Faithfully follows Croce & Hein (ICML 2020) with INR-specific adaptations.

Key adaptations from original APGD:
  - Input: WeightSpaceFeatures (INR weights) instead of images
  - Projection: ICOP (iterative image-space constraint) instead of clamp(0,1) + Linf ball
  - Gradient: computed through unflattener -> classifier w.r.t. flattened weights
  - No clamp(0,1) since INR weights are unbounded
  
Preserved from original:
  - Adaptive step size scheduling with oscillation detection
  - Momentum (alpha=0.75)
  - Loss tracking with checkpoints
  - CE and DLR loss support
  - Multi-restart with subsetting to still-robust samples
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodels.utils.nfn.nfn.layers import FlattenWeights, UnflattenWeights
from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures, params_to_func_params


class APGDAttackINR:
    """
    Auto-PGD for INR weight space with ICOP projection.
    
    Args:
        model:        classifier (victim_wrapper) that takes WeightSpaceFeatures
        renderer:     function that renders INR params to images (from get_batch_inr)
        eps:          perturbation bound in image space (Linf, already scaled for [-1,1])
        n_iter:       number of PGD iterations per restart
        n_restarts:   number of random restarts
        loss:         loss function ('ce' or 'dlr')
        rho:          threshold for step size reduction (default 0.75)
        seed:         random seed
        verbose:      print progress
        device:       torch device
    """

    def __init__(self, model, renderer, eps, n_iter=100, n_restarts=5,
                 loss='ce', rho=0.75, seed=0, verbose=False, device='cuda'):
        self.model = model
        self.renderer = renderer
        self.eps = eps
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.loss = loss
        self.thr_decr = rho
        self.seed = seed
        self.verbose = verbose
        self.device = device
        self.name = f'apgd-{loss}'

        # Checkpoint schedule for step size adaptation (from original APGD)
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    # ------------------------------------------------------------------ #
    #  Flatten / unflatten / render / ICOP helpers                         #
    # ------------------------------------------------------------------ #

    def _init_flatten(self, wsfeat):
        """Initialize flattener/unflattener from a WeightSpaceFeatures sample."""
        net_spec = network_spec_from_wsfeat(wsfeat, set_all_dims=True)
        self._flattener = FlattenWeights(net_spec)
        self._unflattener = UnflattenWeights(net_spec)

    def _flatten(self, wsfeat):
        """Flatten WeightSpaceFeatures to [B, D] tensor."""
        flat_raw = self._flattener(wsfeat)  # [B, D, 1]
        return flat_raw.reshape(flat_raw.shape[0], -1)  # [B, D]

    def _unflatten(self, flat_2d):
        """Reshape [B, D] back to WeightSpaceFeatures."""
        return self._unflattener(flat_2d.unsqueeze(-1))  # [B, D, 1]

    def _render(self, flat_2d):
        """Render images from flattened INR weights."""
        wsfeat = self._unflatten(flat_2d)
        func_params = params_to_func_params(wsfeat)
        return self.renderer(func_params)

    @torch.no_grad()
    def _project_icop(self, adv_flat, orig_flat, I_orig, max_iter=30):
        """
        ICOP projection: iteratively shrink weight perturbation
        until rendered image is within eps of original in Linf.
        """
        delta = adv_flat - orig_flat
        B = delta.size(0)

        for _ in range(max_iter):
            I_adv = self._render(orig_flat + delta)
            diff = (I_adv - I_orig).abs().reshape(B, -1)
            d = diff.max(dim=1).values

            mask = (d > self.eps)
            if not mask.any():
                break

            shrink = (0.9 * self.eps / (d + 1e-12)).clamp(max=1.0)
            shrink = torch.where(mask, shrink, torch.ones_like(shrink))
            delta = delta * shrink.unsqueeze(1)

        # Final safety check: zero out still-violating samples
        I_adv = self._render(orig_flat + delta)
        diff = (I_adv - I_orig).abs().reshape(B, -1)
        d = diff.max(dim=1).values
        still_bad = (d > self.eps)
        if still_bad.any():
            delta[still_bad] = 0.0

        return orig_flat + delta

    # ------------------------------------------------------------------ #
    #  Loss functions                                                      #
    # ------------------------------------------------------------------ #

    def _dlr_loss(self, logits, y):
        """DLR loss (Croce & Hein, 2020). Returns per-sample loss."""
        x_sorted, ind_sorted = logits.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(logits.shape[0])
        # If the correct class has the highest logit, use 2nd highest as comparator
        # Otherwise use the highest
        return -(logits[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    # ------------------------------------------------------------------ #
    #  Step size scheduling                                                #
    # ------------------------------------------------------------------ #

    def _check_oscillation(self, loss_steps, j, k, loss_best, k3=0.75):
        """Check if loss is oscillating (from original APGD)."""
        t = torch.zeros(loss_steps.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (loss_steps[j - counter5] > loss_steps[j - counter5 - 1]).float()
        return (t <= k * k3 * torch.ones_like(t)).float()

    # ------------------------------------------------------------------ #
    #  Core attack                                                         #
    # ------------------------------------------------------------------ #

    def _get_grad_and_logits(self, x_adv_flat, y, criterion_indiv):
        """Compute gradient of loss w.r.t. flattened weights."""
        x_adv_flat = x_adv_flat.detach().requires_grad_(True)
        wsfeat = self._unflatten(x_adv_flat)
        logits = self.model(wsfeat)
        loss_indiv = criterion_indiv(logits, y)
        loss = loss_indiv.sum()
        grad = torch.autograd.grad(loss, x_adv_flat)[0].detach()
        return grad, logits.detach(), loss_indiv.detach()

    def attack_single_run(self, x_flat, y, orig_flat, I_orig, x_init=None):
        """
        Single APGD run in INR weight space.

        Args:
            x_flat:    [B, D] starting point (ignored if x_init is provided)
            y:         [B] labels
            orig_flat: [B, D] original clean weights (projection center)
            I_orig:    [B, ...] rendered original images (for ICOP)
            x_init:    [B, D] optional custom initialization

        Returns:
            (x_best, acc, loss_best, x_best_adv)
        """
        B, D = orig_flat.shape

        # Setup loss function
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self._dlr_loss
        else:
            raise ValueError(f'Unknown loss: {self.loss}')

        # ---- Initialization ---- #
        if x_init is not None:
            x_adv = x_init.clone().detach()
        else:
            # Random initialization within ICOP constraint
            weight_std = orig_flat.std(dim=0, keepdim=True)  # [1, D]
            noise = (2 * torch.rand(B, D, device=self.device) - 1) * weight_std * 0.2
            x_adv = self._project_icop(orig_flat + noise, orig_flat, I_orig)

        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()

        # ---- Initial gradient ---- #
        grad, logits, loss_indiv = self._get_grad_and_logits(x_adv, y, criterion_indiv)
        grad_best = grad.clone()

        acc = logits.max(1)[1] == y
        loss_best = loss_indiv.clone()

        # ---- Tracking arrays ---- #
        loss_steps = torch.zeros([self.n_iter, B], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, B], device=self.device)
        loss_best_steps[0] = loss_best.clone()

        # ---- Step size ---- #
        # Scale initial step size by weight statistics
        weight_scale = orig_flat.std(dim=0).mean()
        step_size = 2.0 * weight_scale * 0.2 * torch.ones([B, 1], device=self.device)

        x_adv_old = x_adv.clone()
        counter3 = 0
        k = self.n_iter_2 + 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        # ---- Main PGD loop ---- #
        for i in range(self.n_iter):
            # === Gradient step with momentum === #
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old  # momentum term
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                # Step: sign of gradient (Linf-style)
                x_adv_1 = x_adv + step_size * grad.sign()
                # ICOP projection
                x_adv_1 = self._project_icop(x_adv_1, orig_flat, I_orig)

                # Apply momentum
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                # ICOP projection again
                x_adv_1 = self._project_icop(x_adv_1, orig_flat, I_orig)

                x_adv = x_adv_1.clone()

            # === Compute new gradient === #
            grad, logits, loss_indiv = self._get_grad_and_logits(x_adv, y, criterion_indiv)

            pred = logits.max(1)[1] == y
            acc = torch.min(acc, pred)
            # Save adversarial examples that cause misclassification
            ind_pred = (pred == 0).nonzero().squeeze()
            if ind_pred.numel() > 0:
                x_best_adv[ind_pred] = x_adv[ind_pred].detach().clone()

            # === Step size scheduling === #
            with torch.no_grad():
                y1 = loss_indiv.clone()
                loss_steps[i] = y1

                # Update best
                ind = (y1 > loss_best).nonzero().squeeze()
                if ind.numel() > 0:
                    x_best[ind] = x_adv[ind].detach().clone()
                    grad_best[ind] = grad[ind].clone()
                    loss_best[ind] = y1[ind]
                loss_best_steps[i + 1] = loss_best.clone()

                counter3 += 1

                if counter3 == k:
                    # Check for oscillation
                    fl_oscillation = self._check_oscillation(
                        loss_steps, i, k, loss_best, k3=self.thr_decr)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                        loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        # Revert to best point for oscillating samples
                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - self.size_decr, self.n_iter_min)
                    counter3 = 0

            if self.verbose and i % 20 == 0:
                print(f'  [{self.name}] iter {i}/{self.n_iter} - '
                      f'best loss: {loss_best.sum():.4f} - '
                      f'robust acc: {acc.float().mean():.2%} - '
                      f'step_size: {step_size.mean():.5f}')
                
        with torch.no_grad():
            I_adv = self._render(x_best_adv)
            diff = (I_adv - I_orig).abs().reshape(B, -1).max(dim=1).values
            n_violations = (diff > self.eps).sum().item()
            print(f'  [ICOP check] max diff: {diff.max():.6f}, '
                f'eps: {self.eps:.6f}, violations: {n_violations}/{B}')
            
        return x_best, acc, loss_best, x_best_adv

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def perturb(self, wsfeat, y):
        """
        Run APGD attack with multiple restarts.
        Only attacks correctly classified samples.
        
        Args:
            wsfeat: WeightSpaceFeatures
            y:      labels [B]

        Returns:
            Adversarial WeightSpaceFeatures
        """
        y = y.detach().clone().long().to(self.device)

        # Initialize flatten/unflatten
        self._init_flatten(wsfeat)
        orig_flat = self._flatten(wsfeat).detach().to(self.device)
        B, D = orig_flat.shape

        # Render original images for ICOP
        with torch.no_grad():
            I_orig = self._render(orig_flat).detach()

        # Check initial accuracy
        with torch.no_grad():
            logits = self.model(wsfeat)
            y_pred = logits.max(1)[1]
        acc = (y_pred == y)

        adv_flat = orig_flat.clone()

        if self.verbose:
            print(f'[{self.name}] initial accuracy: {acc.float().mean():.2%}')

        startt = time.time()
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for counter in range(self.n_restarts):
            # Subset to still-robust samples
            ind_to_fool = acc.nonzero().squeeze()
            if ind_to_fool.dim() == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() == 0:
                break

            x_sub = orig_flat[ind_to_fool].clone()
            y_sub = y[ind_to_fool].clone()
            I_sub = I_orig[ind_to_fool].clone()

            # Run single attack
            _, acc_curr, _, adv_curr = self.attack_single_run(
                x_sub, y_sub, x_sub, I_sub)

            # Update: mark newly fooled samples
            ind_curr = (acc_curr == 0).nonzero().squeeze()
            if ind_curr.dim() == 0:
                ind_curr = ind_curr.unsqueeze(0)
            if ind_curr.numel() > 0:
                acc[ind_to_fool[ind_curr]] = 0
                adv_flat[ind_to_fool[ind_curr]] = adv_curr[ind_curr].detach().clone()

            if self.verbose:
                print(f'  [{self.name}] restart {counter} - '
                      f'robust acc: {acc.float().mean():.2%} - '
                      f'time: {time.time() - startt:.1f}s')

        # Convert back to WeightSpaceFeatures
        return self._unflatten(adv_flat)