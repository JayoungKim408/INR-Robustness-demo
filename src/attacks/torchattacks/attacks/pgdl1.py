import torch
import torch.nn as nn

from ..attack import Attack
from attacks.utils import assert_within_eps


class PGDL1(Attack):
    """
    Distance Measure : L1

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 10.0)
        alpha (float): step size. (Default: 2.0)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL1(model, eps=10.0, alpha=2.0, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model=None,
        eps=10.0,
        alpha=2.0,
        steps=10,
        img_scale=[0, 1], 
        random_start=True,
        eps_for_division=1e-10,
    ):
        super().__init__("PGDL1", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.img_scale = img_scale
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

    def _project_l1_ball(self, delta, eps):
        """
        Project delta onto L1 ball of radius eps.
        Uses the algorithm from Duchi et al. (2008).
        """
        batch_size = delta.size(0)
        original_shape = delta.shape
        # Use reshape instead of view for non-contiguous tensors
        delta_flat = delta.reshape(batch_size, -1)

        # Check if already inside the ball
        l1_norms = torch.norm(delta_flat, p=1, dim=1)
        mask = l1_norms > eps

        if not mask.any():
            return delta

        # Project samples that exceed the budget
        delta_abs = delta_flat[mask].abs()

        # Sort in descending order
        sorted_delta, _ = torch.sort(delta_abs, dim=1, descending=True)

        # Compute cumulative sum
        # Temporarily disable deterministic algorithms for cumsum (not supported on CUDA)
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        if prev_deterministic:
            torch.use_deterministic_algorithms(False, warn_only=True)
        cumsum = torch.cumsum(sorted_delta, dim=1)
        if prev_deterministic:
            torch.use_deterministic_algorithms(True)

        # Find the threshold
        n_dims = delta_flat.size(1)
        rho = torch.arange(1, n_dims + 1, device=delta.device).float()
        threshold = (cumsum - eps) / rho

        # Find largest rho where sorted_delta > threshold
        condition = sorted_delta > threshold
        # Get the index of the last True value in each row
        rho_star = condition.sum(dim=1)

        # Compute lambda (threshold value)
        batch_indices = torch.arange(delta_abs.size(0), device=delta.device)
        lambda_star = threshold[batch_indices, rho_star - 1]

        # Soft thresholding
        delta_proj = delta_flat[mask].sign() * torch.clamp(
            delta_flat[mask].abs() - lambda_star.view(-1, 1), min=0
        )

        # Put back projected values
        result = delta_flat.clone()
        result[mask] = delta_proj

        # Use reshape instead of view_as for non-contiguous tensors
        return result.reshape(original_shape)

    def forward(self, images, labels, model=None):
        r"""
        Overridden.
        """

        if model is not None: # lazy set model
            self.set_model(model)
            self.set_device(next(model.parameters()).device)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Random initialization within L1 ball
            # Sample from Laplacian and scale to be within eps
            delta = torch.empty_like(adv_images).uniform_(-1, 1)
            delta = delta.sign() * torch.empty_like(delta).exponential_()
            d_flat = delta.view(batch_size, -1)
            n = d_flat.norm(p=1, dim=1).view(batch_size, 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta = delta * r / (n + self.eps_for_division) * self.eps
            adv_images = torch.clamp(adv_images + delta, min=self.img_scale[0], max=self.img_scale[1]).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            # L1 steepest descent: move in direction of largest gradient component
            grad_flat = grad.view(batch_size, -1)
            grad_abs = grad_flat.abs()

            # Get the index of max absolute gradient for each sample
            max_idx = grad_abs.argmax(dim=1)

            # Create sparse update (only update the coordinate with max gradient)
            update = torch.zeros_like(grad_flat)
            batch_indices = torch.arange(batch_size, device=self.device)
            update[batch_indices, max_idx] = grad_flat[batch_indices, max_idx].sign() * self.alpha
            update = update.view_as(grad)

            adv_images = adv_images.detach() + update

            # Project onto L1 ball
            delta = adv_images - images
            delta = self._project_l1_ball(delta, self.eps)

            adv_images = torch.clamp(images + delta, min=self.img_scale[0], max=self.img_scale[1]).detach()
            
        self.check_projection(images, adv_images)
        return adv_images


    def check_projection(self, imgs, adv_imgs):
        report, stats, arrays = assert_within_eps(imgs, adv_imgs, self.eps, norm='l1')
        print("projection check:", report)
