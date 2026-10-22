from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from ..attack import Attack


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss functions (from utils/criterion.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _dlr_loss(x, y):
    """DLR loss (untargeted).

    DLR = -(z_y - max_{i!=y} z_i) / (z_{pi_1} - z_{pi_3})
    """
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    z_y = x[np.arange(x.shape[0]), y]
    max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    value_true_maximum = z_y - max_zi
    pi1_pi3 = x_sorted[:, -1] - x_sorted[:, -3] + 1e-6
    loss = (-1.0) * value_true_maximum / pi1_pi3
    return loss.reshape(-1)


def _cw_loss(x, y):
    """CW loss (untargeted).

    CW = -(z_y - max_{i!=y} z_i)
    """
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    z_y = x[np.arange(x.shape[0]), y]
    max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum
    return loss.reshape(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Conjugate gradient helpers (from AUTOConjugate.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_beta(gradn, gradn_1, sn_1, method="HS", use_clamp=True,
              replace_nan=False):
    """Compute conjugate gradient beta coefficient.

    Supports: FR, PR, HS, DY, HZ, DL, LS, RMIL, RMIL+.
    """
    batch_size = gradn.shape[0]
    gradn = -gradn.reshape(batch_size, -1)
    gradn_1 = -gradn_1.reshape(batch_size, -1)
    sn_1 = sn_1.reshape(batch_size, -1)
    y = gradn - gradn_1

    if method == "FR":
        betan = gradn.pow(2).sum(dim=1) / (gradn_1.pow(2).sum(dim=1))
    elif method == "PR":
        betan = (gradn * y).sum(dim=1) / (gradn_1.pow(2).sum(dim=1))
    elif method == "HS":
        betan = -(gradn * y).sum(dim=1) / ((sn_1 * y).sum(dim=1))
    elif method == "DY":
        betan = -gradn.pow(2).sum(dim=1) / ((sn_1 * y).sum(dim=1))
    elif method == "HZ":
        betan = (
            (
                y
                - 2
                * sn_1
                * (
                    y.pow(2).sum(dim=1) / ((sn_1 * y).sum(dim=1)).unsqueeze(-1)
                )
            )
            * gradn
        ).sum(dim=1) / ((sn_1 * y).sum(dim=1))
    elif method == "DL":
        betan = ((y - sn_1) * gradn).sum(dim=1) / (
            (sn_1 * y).sum(dim=1)
        )
    elif method == "LS":
        betan = -(gradn * y).sum(dim=1) / ((sn_1 * gradn_1).sum(dim=1))
    elif method == "RMIL":
        betan = (gradn * y).sum(dim=1) / (sn_1.pow(2).sum(dim=1))
    elif method == "RMIL+":
        betan = (gradn * (y - sn_1)).sum(dim=1) / (
            sn_1.pow(2).sum(dim=1)
        )
    else:
        raise NotImplementedError(f"Unknown beta method: {method}")

    if use_clamp:
        betan = betan.clamp(min=0)
    if replace_nan:
        betan[torch.isnan(betan)] = 0.0
    return betan


# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive step size (from utils/step_size.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_oscillation(x, j, k, y5, k3=0.75):
    """Check if loss oscillates over the last k iterations.

    Returns a boolean mask (per sample) where True means oscillation detected.
    """
    t = np.zeros(x.shape[1])
    for counter5 in range(k):
        t += x[j - counter5] > x[j - counter5 - 1]
    return t <= k * k3 * np.ones(t.shape)


def _update_step_size(
    u, xk, x_best, grad, gradn_1, sn_1,
    grad_best, grad_best_1, s_best_1,
    step_size, iteration, counter3, k,
    loss_steps, loss_best, loss_best_last_check,
    reduced_last_check, thr_decr, size_decr, n_iter_min,
    activate_flag=False,
):
    """APGD-style adaptive step size update.

    At every checkpoint (counter3 == k), checks for oscillation and
    lack of improvement, halving step size for flagged samples and
    reverting them to their best-so-far state.
    """
    if counter3 == k:
        fl_oscillation = _check_oscillation(
            loss_steps.detach().cpu().numpy(),
            iteration,
            k,
            loss_best.detach().cpu().numpy(),
            k3=thr_decr,
        )
        did_not_move = (
            (grad == gradn_1).all(dim=1).all(dim=1).all(dim=1)
            .cpu().numpy() * activate_flag
        )
        fl_oscillation = ((fl_oscillation + did_not_move) / 2).astype(bool)
        fl_reduce_no_impr = (~reduced_last_check) * (
            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
        )
        fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
        reduced_last_check = np.copy(fl_oscillation)
        loss_best_last_check = loss_best.clone()

        if np.sum(fl_oscillation) > 0:
            step_size[u[fl_oscillation]] /= 2.0
            fl_oscillation = np.where(fl_oscillation)
            xk[fl_oscillation] = x_best[fl_oscillation].clone()
            grad[fl_oscillation] = grad_best[fl_oscillation].clone()
            gradn_1[fl_oscillation] = grad_best_1[fl_oscillation].clone()
            sn_1[fl_oscillation] = s_best_1[fl_oscillation].clone()

        k = np.maximum(k - size_decr, n_iter_min)
        counter3 = 0

    return (
        u, xk, x_best, grad, gradn_1, sn_1,
        step_size, counter3, k,
        loss_steps, loss_best, loss_best_last_check, reduced_last_check,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTOConjugate Attack
# ═══════════════════════════════════════════════════════════════════════════════

class AUTOConjugate(Attack):
    r"""
    Auto-Conjugate Gradient attack (ACG).

    Conjugate-gradient variant of Auto-PGD that uses nonlinear conjugate
    gradient directions (e.g. Hestenes–Stiefel) instead of plain sign-gradient,
    combined with APGD's adaptive step-size schedule.

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation (L-inf). (Default: 8/255)
        steps (int): number of iterations per restart. (Default: 100)
        num_restarts (int): number of random restarts. (Default: 1)
        eot_iter (int): number of EOT iterations for gradient averaging. (Default: 1)
        thr_decr (float): rho — oscillation threshold for step-size halving. (Default: 0.75)
        beta_method (str): conjugate gradient beta formula.
            One of 'FR', 'PR', 'HS', 'DY', 'HZ', 'DL', 'LS', 'RMIL', 'RMIL+'.
            (Default: 'HS')
        momentum_alpha (float): momentum interpolation coefficient (a). (Default: 0.75)
        use_clamp (bool): clamp beta to >= 0. (Default: True)
        replace_nan (bool): replace NaN betas with 0. (Default: False)
        activate_flag (bool): enable the "did not move" check in step-size
            update. (Default: False)
        criterion (str): loss function — 'dlr', 'ce', or 'cw'. (Default: 'dlr')
        initial_point (str): starting point for restart 0 — 'input' or 'random'.
            (Default: 'input')
        restart_point (str): starting point for restarts > 0 — 'best' or 'random'.
            (Default: 'random')
        img_scale (list): [min, max] pixel range. (Default: [0, 1])

    Note:
        - This is an **untargeted** attack; targeted mode is not supported.
        - The restart loop uses CW loss to select the best adversarial example
          across restarts (matching the original ``RestartAttacker._attack``).
        - Random initialization uses the standard APGD formula:
          ``x_nat + eps * t / max|t|`` (L-inf normalized), but without the
          per-image deterministic seeding of the original. Given the same
          initial point, all subsequent iterations are numerically identical.

    Examples::
        >>> attack = AUTOConjugate(model, eps=8/255, steps=100, num_restarts=1)
        >>> adv_images = attack(images, labels)
    """

    def __init__(
        self,
        model=None,
        eps=8 / 255,
        steps=100,
        num_restarts=1,
        eot_iter=1,
        thr_decr=0.75,
        beta_method="HS",
        momentum_alpha=0.75,
        use_clamp=True,
        replace_nan=False,
        activate_flag=False,
        criterion="dlr",
        initial_point="input",
        restart_point="random",
        img_scale=(0, 1),
    ):
        super().__init__("AUTOConjugate", model)
        self.eps = eps
        self.steps = steps
        self.num_restarts = num_restarts
        self.eot_iter = eot_iter
        self.thr_decr = thr_decr
        self.beta_method = beta_method
        self.momentum_alpha = momentum_alpha
        self.use_clamp = use_clamp
        self.replace_nan = replace_nan
        self.activate_flag = activate_flag
        self.criterion_name = criterion
        self.initial_point = initial_point
        self.restart_point = restart_point
        self.img_scale = img_scale
        self.supported_mode = ["default"]

    # ── loss selection ────────────────────────────────────────────────────

    def _get_criterion(self):
        """Return the per-sample loss function used inside the attack loop."""
        if self.criterion_name == "dlr":
            return _dlr_loss
        elif self.criterion_name == "ce":
            ce = nn.CrossEntropyLoss(reduce=False, reduction="mean")
            return ce
        elif self.criterion_name == "cw":
            return _cw_loss
        else:
            raise ValueError(f"Unknown criterion: {self.criterion_name}")

    # ── per-element projection onto L-inf ball ∩ [0,1] ───────────────────

    @staticmethod
    def _clamp(x_adv, lower, upper):
        """Project onto the intersection of the L-inf ball and pixel range.

        Equivalent to the original ``Clamp.__call__``.
        """
        return torch.minimum(torch.maximum(x_adv, lower), upper)

    # ── APGD-style random start (from getRandomInput4AutoPGD) ────────────

    def _random_start(self, x_nat):
        """Generate a random starting point on the L-inf sphere.

        ``x_adv = x_nat + eps * t / ||t||_inf``, clamped to [0, 1].
        Numerically identical to the original ``getRandomInput4AutoPGD``
        (modulo RNG seed; the original uses per-image deterministic seeds).
        """
        t = 2 * torch.rand_like(x_nat) - 1
        # Normalize by per-sample L-inf norm so perturbation has L-inf = eps
        t_linf = (
            t.reshape(t.shape[0], -1)
            .abs()
            .max(dim=1, keepdim=True)[0]
            .reshape(-1, *([1] * len(x_nat.shape[1:])))
        )
        x_adv = x_nat.detach() + self.eps * torch.ones(
            [x_nat.shape[0], *([1] * len(x_nat.shape[1:]))],
            device=x_nat.device,
        ).detach() * t / t_linf
        x_adv = x_adv.clamp(self.img_scale[0], self.img_scale[1])
        return x_adv

    # ── single run (from _attack_single_run_fast) ────────────────────────

    @torch.no_grad()
    def _attack_single_run(self, x_nat, y, x_prev_best, restart):
        """Single-restart conjugate gradient attack.

        Line-for-line equivalent to the original
        ``AUTOConjugate._attack_single_run_fast``, with statistics-only
        tensors removed.

        Arguments:
            x_nat: clean images (on device).
            y: ground-truth labels (on device).
            x_prev_best: best adversarial examples from previous restarts (on device).
            restart: current restart index (int).

        Returns:
            x_best: best adversarial examples found in this run (on CPU).
        """
        criterion = self._get_criterion()
        device = self.device
        bs = x_nat.shape[0]

        n_iter_2 = max(int(0.22 * self.steps), 1)
        n_iter_min = max(int(0.06 * self.steps), 1)
        size_decr = max(int(0.03 * self.steps), 1)

        # ── bounds ────────────────────────────────────────────────────────
        lower = torch.clamp(x_nat - self.eps, self.img_scale[0], self.img_scale[1]).to(device)
        upper = torch.clamp(x_nat + self.eps, self.img_scale[0], self.img_scale[1]).to(device)

        # ── initial point ─────────────────────────────────────────────────
        if restart > 0:
            if self.restart_point == "best":
                xk = x_prev_best.clone()
            else:
                xk = self._random_start(x_nat).to(device)
        else:
            if self.initial_point == "input":
                xk = x_nat.clone().to(device)
            else:
                xk = self._random_start(x_nat).to(device)

        x_best = xk.clone()
        x_best_adv = xk.clone()
        loss_steps = torch.zeros([self.steps, bs])

        # ── initial gradient ──────────────────────────────────────────────
        xk.requires_grad_()
        grad = torch.zeros_like(xk)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.get_logits(xk)
                loss_indiv = criterion(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [xk])[0].detach()
        grad /= float(self.eot_iter)

        grad_best = grad.clone()
        grad_best_1 = grad.clone()
        s_best_1 = grad.clone()

        loss_best = loss_indiv.detach().clone()

        # ── step size ─────────────────────────────────────────────────────
        alpha = 2.0
        step_size = (
            alpha
            * self.eps
            * torch.ones(
                [bs, *([1] * len(x_nat.shape[1:]))],
                device=device,
            )
        )

        k = n_iter_2 + 0
        u = torch.arange(bs, device=device)
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # ── main loop ─────────────────────────────────────────────────────
        sn_1 = None
        x_prev = xk.clone()

        for iteration in range(self.steps):
            gradn = grad.clone()
            xk = xk.detach()
            grad2 = xk - x_prev
            x_prev = xk.clone()

            # Conjugate direction
            if iteration == 0:
                sn = gradn
                a = 1.0
            else:
                a = self.momentum_alpha
                beta_n = (
                    _get_beta(
                        gradn, gradn_1, sn_1,
                        method=self.beta_method,
                        use_clamp=self.use_clamp,
                        replace_nan=self.replace_nan,
                    )
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                sn = gradn + beta_n * sn_1

            sn_1 = sn.clone()
            gradn_1 = grad.clone()

            # Step with momentum
            xk_tmp = self._clamp(xk + step_size * torch.sign(sn), lower, upper)
            xk = self._clamp(
                xk + a * (xk_tmp - xk) + (1 - a) * grad2, lower, upper
            ).detach()

            # Gradient at new point
            xk.requires_grad_()
            grad = torch.zeros_like(xk)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.get_logits(xk)
                    loss_indiv = criterion(logits, y)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [xk])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y

            # Track best adversarial (misclassified) image
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                xk[(pred == 0).nonzero().squeeze()] + 0.0
            )

            # Update best by loss
            y1 = loss_indiv.detach().clone()
            loss_steps[iteration] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = xk[ind].clone()
            grad_best[ind] = grad[ind].clone()
            grad_best_1[ind] = gradn_1[ind].clone()
            s_best_1[ind] = sn_1[ind].clone()
            loss_best[ind] = y1[ind] + 0

            # Adaptive step size
            counter3 += 1
            (
                u, xk, x_best, grad, gradn_1, sn_1,
                step_size, counter3, k,
                loss_steps, loss_best, loss_best_last_check, reduced_last_check,
            ) = _update_step_size(
                u=u, xk=xk, x_best=x_best,
                grad=grad, gradn_1=gradn_1, sn_1=sn_1,
                grad_best=grad_best, grad_best_1=grad_best_1, s_best_1=s_best_1,
                step_size=step_size,
                iteration=iteration, counter3=counter3, k=k,
                loss_steps=loss_steps, loss_best=loss_best,
                loss_best_last_check=loss_best_last_check,
                reduced_last_check=reduced_last_check,
                thr_decr=self.thr_decr, size_decr=size_decr,
                n_iter_min=n_iter_min,
                activate_flag=self.activate_flag,
            )

        return x_best.cpu()

    # ── forward (multi-restart, from RestartAttacker._attack) ─────────────

    def forward(self, images, labels, model=None):
        r"""
        Overridden.
        """
        if model is not None:
            self.set_model(model)
            self.set_device(next(model.parameters()).device)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # ── initial evaluation ────────────────────────────────────────────
        adv_images = images.clone()
        bs = images.shape[0]

        with torch.no_grad():
            logits = self.get_logits(adv_images)
        acc = logits.max(dim=1)[1] == labels  # True = correctly classified

        # CW loss for cross-restart comparison (higher = better attack)
        best_eval = _cw_loss(logits, labels)

        # ── restart loop ──────────────────────────────────────────────────
        for restart in range(self.num_restarts):
            if acc.sum() == 0:
                break

            xk = adv_images[acc].clone()
            yk = labels[acc].clone()
            x_nat = images[acc].clone()

            x_best_run = self._attack_single_run(
                x_nat=x_nat,
                y=yk,
                x_prev_best=xk,
                restart=restart,
            ).to(self.device)

            # Evaluate this restart's result
            with torch.no_grad():
                logits_run = self.get_logits(x_best_run)
            acc_run = logits_run.max(dim=1)[1] == yk
            eval_run = _cw_loss(logits_run, yk)

            # Keep per-sample best across restarts (by CW loss)
            is_better = eval_run > best_eval[acc]
            tmp_adv = adv_images[acc]
            tmp_adv[is_better] = x_best_run[is_better].clone()
            adv_images[acc] = tmp_adv.clone()

            tmp_eval = best_eval[acc]
            tmp_eval[is_better] = eval_run[is_better].clone()
            best_eval[acc] = tmp_eval.clone()

            # Remove successfully attacked samples from future restarts
            acc_clone = acc.clone()
            tmp_acc = acc_clone[acc_clone].clone()
            tmp_acc[~acc_run] = False
            acc[acc_clone] = tmp_acc.clone()

        return adv_images