import torch
import torch.nn as nn

from ..attack import Attack
from basemodels.utils.nfn.nfn.layers import FlattenWeights, UnflattenWeights
from basemodels.utils.nfn.nfn.common import network_spec_from_wsfeat, params_to_state_dicts, params_to_func_params
from attacks.utils import assert_within_eps

class PGDWSFeat(Attack):

    def __init__(
        self,
        model=None,
        eps: float = 0.01,
        alpha: float = 0.005,
        steps: int = 10,
        random_start: bool = True,
        proj_mode: str = "icop",
        renderer=None,
        eps_img: float | None = None,
    ):
        super().__init__("PGDWSFeat", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

        assert proj_mode in ("weight_linf", "image_linf", "icop")
        self.proj_mode = proj_mode

        self.renderer = renderer
        self.eps_img = eps_img
        
    def _compute_linf_img_grad(self, adv_flat, unflattener, I_orig):
        # Build adv image (requires grad)
        adv_wsfeat = unflattener(adv_flat)
        params = params_to_func_params(adv_wsfeat)
        I_adv = self.renderer(params)    # (B, C, H, W), requires_grad = True

        # Elementwise abs diff
        diff = (I_adv - I_orig).abs()    # (B, C, H, W)

        # Find pixel(s) contributing to max
        # diff_flat: (B, C*H*W)
        diff_flat = diff.reshape(diff.size(0), -1)
        idx = diff_flat.argmax(dim=1)    # (B,)

        # Build one-hot mask for max positions
        B, C, H, W = diff.shape
        mask = torch.zeros_like(diff_flat)
        mask[torch.arange(B), idx] = 1.0
        mask = mask.view(B, C, H, W)     # (B, C, H, W)

        # Multiply mask by sign(I_adv - I_orig) to form subgradient of L∞
        grad_I = mask * torch.sign(I_adv - I_orig)  # (B, C, H, W)

        g_img = torch.autograd.grad(
            I_adv, adv_flat,
            grad_outputs=grad_I,
            retain_graph=False,
            create_graph=False
        )[0]

        return g_img


    @torch.no_grad()
    def _project_image_linf_iterative(self, adv_flat, x_orig, unflattener, I_orig, max_iter=30):
        eps_img = self.eps_img if self.eps_img is not None else self.eps

        delta = adv_flat - x_orig  # (B, D)
        B = delta.size(0)

        for i in range(max_iter):
            adv_wsfeat = unflattener(x_orig + delta)
            params = params_to_func_params(adv_wsfeat)
            I_adv = self.renderer(params)   # (B,C,H,W)

            diff = (I_adv - I_orig).abs()
            diff_flat = diff.reshape(B, -1)
            d = diff_flat.max(dim=1).values   # (B,)

            # mask samples that exceed eps bound
            mask = (d > eps_img)
            if not mask.any():
                break

            # Conservative shrink factor (0.9x) to account for non-linearity
            shrink = (0.9 * eps_img / (d + 1e-12)).clamp(max=1.0)   # (B,)

            # Only shrink violating samples
            shrink = torch.where(mask, shrink, torch.ones_like(shrink))

            # reshape for broadcast
            shrink = shrink.view(-1, *([1] * (delta.dim() - 1)))
            delta = delta * shrink

        # Final check: if any sample still violates, reset to x_orig (delta=0)
        adv_wsfeat = unflattener(x_orig + delta)
        params = params_to_func_params(adv_wsfeat)
        I_adv = self.renderer(params)
        diff = (I_adv - I_orig).abs().reshape(B, -1)
        d = diff.max(dim=1).values
        still_violating = (d > eps_img)

        if still_violating.any():
            # Zero out delta for samples that still violate (guaranteed to satisfy bound)
            zero_mask = still_violating.view(-1, *([1] * (delta.dim() - 1)))
            delta = torch.where(zero_mask, torch.zeros_like(delta), delta)

        return (x_orig + delta).detach()


    def forward(self, wsfeat, labels, model=None):
        """
        wsfeat: WeightSpaceFeatures 타입 (구조화된 INR weight)
        labels: (N,)
        """
        if model is not None: # lazy set model
            self.set_model(model)
            self.set_device(next(model.parameters()).device)

        labels = labels.clone().detach().to(self.device)

        net_spec = network_spec_from_wsfeat(wsfeat, set_all_dims=True)
        flattener = FlattenWeights(net_spec)
        unflattener = UnflattenWeights(net_spec)

        x_flat = flattener(wsfeat).detach().to(self.device)  # (B, D, ...)
        x_orig = x_flat.clone().detach()                     # projection center

        if self.targeted:
            target_labels = self.get_target_label(wsfeat, labels)

        loss_fn = nn.CrossEntropyLoss()

        if self.proj_mode in ("image_linf", "icop"):
            with torch.no_grad():
                wsfeat_orig = unflattener(x_orig)
                state_dicts = params_to_func_params(wsfeat_orig)
                I_orig = self.renderer(state_dicts).detach()
        else:
            I_orig = None

        adv_flat = x_orig.clone().detach()

        # if self.random_start:
        #     adv_flat = adv_flat + torch.empty_like(adv_flat).uniform_(-self.eps, self.eps)
        #     adv_flat = adv_flat.detach()

        #     adv_flat = self._project_image_linf_iterative(adv_flat, x_orig, unflattener, I_orig, max_iter=10)
        if self.random_start:
            if self.proj_mode in ("image_linf", "icop"):
                # image space 기준으로 작은 랜덤 초기화
                noise_scale = (self.eps_img if self.eps_img is not None else self.eps) * 0.1
                adv_flat = adv_flat + torch.empty_like(adv_flat).uniform_(-noise_scale, noise_scale)
            else:
                adv_flat = adv_flat + torch.empty_like(adv_flat).uniform_(-self.eps, self.eps)
            adv_flat = adv_flat.detach()

            if self.proj_mode in ("image_linf", "icop"):
                adv_flat = self._project_image_linf_iterative(adv_flat, x_orig, unflattener, I_orig, max_iter=10)

        # ----- PGD loop -----
        with torch.enable_grad():
            for _ in range(self.steps):
                adv_flat.requires_grad = True

                adv_wsfeat = unflattener(adv_flat)
                logits = self.get_logits(adv_wsfeat)

                if self.targeted:
                    cost = -loss_fn(logits, target_labels)
                else:
                    cost = loss_fn(logits, labels)

                if self.proj_mode == "icop":
                    g_cls = torch.autograd.grad(
                        cost, adv_flat,
                        retain_graph=True,
                        create_graph=False
                    )[0]

                    g_img = self._compute_linf_img_grad(
                        adv_flat, unflattener, I_orig
                    )

                    Bsz = g_cls.size(0)
                    g_cls_flat = g_cls.reshape(Bsz, -1)
                    g_img_flat = g_img.reshape(Bsz, -1)

                    dot = (g_cls_flat * g_img_flat).sum(dim=1, keepdim=True)
                    norm_sq = (g_img_flat**2).sum(dim=1, keepdim=True) + 1e-12

                    coeff = dot / norm_sq
                    g_orth_flat = g_cls_flat - coeff * g_img_flat
                    g = g_orth_flat.view_as(g_cls)
                else:

                    g = torch.autograd.grad(
                        cost,
                        adv_flat,
                        retain_graph=False,
                        create_graph=False,
                    )[0]

                adv_flat = adv_flat.detach() + self.alpha * g.sign()

                adv_flat = self._project_image_linf_iterative(
                    adv_flat, x_orig, unflattener, I_orig, max_iter=10
                )

        with torch.no_grad():
            adv_wsfeat = unflattener(adv_flat)
            state_dicts = params_to_func_params(adv_wsfeat)
            I_adv = self.renderer(state_dicts)

        self.check_projection(I_orig, I_adv)
        adv_wsfeat = unflattener(adv_flat)
        return adv_wsfeat
    

    def check_projection(self, imgs, adv_imgs):
        report, stats, arrays = assert_within_eps(imgs, adv_imgs, self.eps_img, norm='linf')
        # print("projection check:", report)
