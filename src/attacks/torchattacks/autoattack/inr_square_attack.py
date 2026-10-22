"""
Square Attack in INR weight space with ICOP projection.
Gradient-free, query-based black-box attack on INR-based classifiers.
No surrogate or INR initialization needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from basemodels.utils.nfn.nfn.layers import FlattenWeights, UnflattenWeights
from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, params_to_func_params


class SquareAttackINR:
    """
    Square Attack adapted for INR weight space.
    
    Instead of perturbing square patches in image space,
    perturbs random subsets of INR weight dimensions.
    Uses ICOP projection to ensure image-space constraint.
    
    Args:
        model:       classifier (victim_wrapper) that takes WeightSpaceFeatures
        renderer:    function that renders INR weights to images
        eps:         perturbation bound in image space (Linf)
        n_queries:   max number of queries per restart
        p_init:      initial fraction of dimensions to perturb
        n_restarts:  number of random restarts
        loss:        loss type ('margin' or 'ce')
        verbose:     print progress
        device:      torch device
    """
    def __init__(
            self,
            model,
            renderer,
            eps,
            n_queries=100,
            p_init=0.8,
            n_restarts=1,
            seed=0,
            verbose=False,
            loss='margin',
            resc_schedule=True,
            device='cuda'):
        
        self.model = model
        self.renderer = renderer
        self.eps = eps
        self.n_queries = n_queries
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.device = device
        self._flat_shape = None

    def _flatten_wsfeat(self, wsfeat):
        """Flatten WeightSpaceFeatures to [B, D] tensor."""
        net_spec = network_spec_from_wsfeat(wsfeat, set_all_dims=True)
        flattener = FlattenWeights(net_spec)
        unflattener = UnflattenWeights(net_spec)
        flat_raw = flattener(wsfeat)
        self._flat_shape = flat_raw.shape  # e.g. [B, D, 1]
        flat = flat_raw.reshape(flat_raw.shape[0], -1)  # [B, D]
        return flat, flattener, unflattener, net_spec

    def _unflatten(self, flat_2d, unflattener):
        """Reshape [B, D] back to original shape and unflatten to WeightSpaceFeatures."""
        return unflattener(flat_2d.reshape(self._flat_shape))

    def _render_from_flat(self, flat, unflattener):
        """Render images from flattened INR weights."""
        wsfeat = self._unflatten(flat, unflattener)
        params = params_to_func_params(wsfeat)
        imgs = self.renderer(params)
        return imgs

    @torch.no_grad()
    def _project_icop(self, adv_flat, orig_flat, unflattener, I_orig, max_iter=30):
        """
        ICOP projection: iteratively shrink weight perturbation
        until rendered image is within eps of original.
        """
        delta = adv_flat - orig_flat
        B = delta.size(0)

        for i in range(max_iter):
            I_adv = self._render_from_flat(orig_flat + delta, unflattener)
            diff = (I_adv - I_orig).abs()
            diff_flat = diff.reshape(B, -1)
            d = diff_flat.max(dim=1).values

            mask = (d > self.eps)
            if not mask.any():
                break

            shrink = (0.9 * self.eps / (d + 1e-12)).clamp(max=1.0)
            shrink = torch.where(mask, shrink, torch.ones_like(shrink))
            shrink = shrink.view(-1, *([1] * (delta.dim() - 1)))
            delta = delta * shrink

        # Final check: zero out still-violating samples
        I_adv = self._render_from_flat(orig_flat + delta, unflattener)
        diff = (I_adv - I_orig).abs().reshape(B, -1)
        d = diff.max(dim=1).values
        still_violating = (d > self.eps)
        if still_violating.any():
            zero_mask = still_violating.view(-1, *([1] * (delta.dim() - 1)))
            delta = torch.where(zero_mask, torch.zeros_like(delta), delta)

        return orig_flat + delta

    def margin_and_loss(self, wsfeat, y):
        """Compute margin and loss from WeightSpaceFeatures."""
        logits = self.model(wsfeat)
        xent = F.cross_entropy(logits, y, reduction='none')
        u = torch.arange(logits.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        if self.loss == 'ce':
            return y_corr - y_others, -1. * xent
        elif self.loss == 'margin':
            return y_corr - y_others, y_corr - y_others

    def margin_and_loss_flat(self, flat, unflattener, y):
        """Compute margin and loss from flattened weights."""
        wsfeat = self._unflatten(flat, unflattener)
        return self.margin_and_loss(wsfeat, y)

    def p_selection(self, it):
        """Schedule to decrease the fraction of perturbed dimensions."""
        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init
        return p

    def attack_single_run(self, wsfeat, y):
        """
        Single run of INR weight-space Square Attack.
        Full batch is maintained throughout (no subsetting).
        """
        with torch.no_grad():
            # Flatten
            orig_flat, flattener, unflattener, net_spec = self._flatten_wsfeat(wsfeat)
            orig_flat = orig_flat.detach().to(self.device)
            B, D = orig_flat.shape

            # Render original images for ICOP
            I_orig = self._render_from_flat(orig_flat, unflattener).detach()

            # Estimate perturbation scale from weight statistics
            weight_std = orig_flat.std(dim=0, keepdim=True)  # [1, D]

            # Initial random perturbation
            n_init = max(int(self.p_init * D), 1)
            init_mask = torch.zeros(B, D, device=self.device)
            for b in range(B):
                idx = torch.randperm(D, device=self.device)[:n_init]
                init_mask[b, idx] = 1.0
            
            noise = (2 * torch.rand(B, D, device=self.device) - 1) * weight_std * 0.1
            delta_init = noise * init_mask

            best_flat = self._project_icop(
                orig_flat + delta_init, orig_flat, unflattener, I_orig
            )
            
            margin_min, loss_min = self.margin_and_loss_flat(best_flat, unflattener, y)
            n_queries = torch.ones(B, device=self.device)
            n_ex_total = B

            if (margin_min < 0.0).all():
                return n_queries, self._unflatten(best_flat, unflattener)

            for i_iter in range(self.n_queries):
                active = (margin_min > 0.0)
                if active.sum() == 0:
                    break

                p = self.p_selection(i_iter)
                n_dims = max(int(p * D), 1)

                # Random subset of dimensions to perturb
                dim_mask = torch.zeros(B, D, device=self.device)
                for b in range(B):
                    if active[b]:
                        idx = torch.randperm(D, device=self.device)[:n_dims]
                        dim_mask[b, idx] = 1.0

                # Random perturbation scaled by weight statistics
                noise = (2 * torch.rand(B, D, device=self.device) - 1) * weight_std * 0.1
                new_delta = noise * dim_mask

                # Apply to current best
                candidate_flat = best_flat.clone() + new_delta

                # ICOP projection
                candidate_flat = self._project_icop(
                    candidate_flat, orig_flat, unflattener, I_orig
                )

                # Evaluate
                margin, loss = self.margin_and_loss_flat(candidate_flat, unflattener, y)

                # Update only active samples where loss improved
                idx_improved = (loss < loss_min).float() * active.float()
                idx_miscl = (margin <= 0.).float()
                idx_improved = torch.max(idx_improved, idx_miscl * active.float())

                loss_min = idx_improved * loss + (1. - idx_improved) * loss_min
                margin_min = idx_improved * margin + (1. - idx_improved) * margin_min

                idx_improved_expand = idx_improved.unsqueeze(1)  # [B, 1]
                best_flat = idx_improved_expand * candidate_flat + (1. - idx_improved_expand) * best_flat
                n_queries += active.float()

                ind_succ = (margin_min <= 0.).nonzero().squeeze()
                if self.verbose and ind_succ.numel() != 0 and i_iter % 10 == 0:
                    print(f'{i_iter + 1} - success rate={ind_succ.numel()}/{n_ex_total} '
                          f'({ind_succ.numel()/n_ex_total:.2%}) - '
                          f'avg queries={n_queries[ind_succ].mean().item():.1f} - '
                          f'loss={loss_min.mean():.3f}')

                if ind_succ.numel() == n_ex_total:
                    break

        return n_queries, self._unflatten(best_flat, unflattener)

    def perturb(self, wsfeat, y=None):
        """
        Run INR weight-space Square Attack.
        
        Args:
            wsfeat: WeightSpaceFeatures (fitted INR weights)
            y:      labels
            
        Returns:
            Adversarial WeightSpaceFeatures
        """
        if y is None:
            with torch.no_grad():
                logits = self.model(wsfeat)
                y = logits.max(1)[1].detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        startt = time.time()
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        _, adv_wsfeat = self.attack_single_run(wsfeat, y)

        if self.verbose:
            with torch.no_grad():
                acc = self.model(adv_wsfeat).max(1)[1] == y
                print(f'robust accuracy: {acc.float().mean():.2%} - '
                      f'time: {time.time() - startt:.1f} s')

        return adv_wsfeat