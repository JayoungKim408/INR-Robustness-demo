import torch
import torch.nn as nn

from ..attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    
    Batched implementation — backward is per-class, not per-sample.
    Distance Measure : L2 (minimal perturbation)

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
        img_scale (list): [min, max] of image range. (Default: [0, 1])
    """

    def __init__(self, model=None, steps=50, overshoot=0.02, img_scale=[0, 1]):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.img_scale = img_scale
        self.supported_mode = ["default"]

    def forward(self, images, labels, model=None):
        if model is not None:
            self.set_model(model)
            self.set_device(next(model.parameters()).device)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = images.shape[0]
        adv_images = images.clone().detach()
        active = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.steps):
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]
            active_imgs = adv_images[active_idx].clone().detach().requires_grad_(True)
            active_labels = labels[active_idx]

            # Forward
            logits = self.get_logits(active_imgs)
            preds = logits.argmax(dim=1)

            # Mark fooled
            fooled = preds != active_labels
            if fooled.any():
                active[active_idx[fooled]] = False

            still_active = ~fooled
            if not still_active.any():
                break

            still_idx = still_active.nonzero(as_tuple=True)[0]
            still_global = active_idx[still_idx]
            still_imgs = adv_images[still_global].clone().detach().requires_grad_(True)
            still_labels = labels[still_global]

            # Fresh forward for still-active samples
            still_logits = self.get_logits(still_imgs)
            n = still_imgs.shape[0]
            num_classes = still_logits.shape[1]
            flat_dim = still_imgs[0].numel()

            # Gradient for true class — one backward for entire batch
            f_0 = still_logits[torch.arange(n), still_labels]
            grad_0 = torch.autograd.grad(f_0.sum(), still_imgs, retain_graph=True)[0]
            grad_0_flat = grad_0.reshape(n, -1)  # (n, flat_dim)
            f_0 = f_0.detach() 

            best_value = torch.full((n,), float('inf'), device=self.device)
            best_delta = torch.zeros_like(still_imgs)

            for k in range(num_classes):
                # Skip samples where k == true label
                mask = (k != still_labels)
                if not mask.any():
                    continue

                # Gradient for class k — one backward for entire batch
                f_k = still_logits[:, k]
                grad_k = torch.autograd.grad(f_k.sum(), still_imgs, retain_graph=True)[0]
                grad_k_flat = grad_k.reshape(n, -1)  # (n, flat_dim)
                f_k = f_k.detach()

                # f' and w'
                f_prime = f_k - f_0  # (n,)
                w_prime = grad_k_flat - grad_0_flat  # (n, flat_dim)
                w_norm = torch.norm(w_prime, p=2, dim=1).clamp(min=1e-8)  # (n,)

                value = torch.abs(f_prime) / w_norm  # (n,)

                # Update best where this class is closer
                update = mask & (value < best_value)
                if update.any():
                    best_value[update] = value[update]
                    coeff = (torch.abs(f_prime) / (w_norm ** 2)).unsqueeze(1)  # (n, 1)
                    delta = (coeff * w_prime).view_as(still_imgs)  # (n, C, H, W)
                    best_delta[update] = delta[update]

            # Apply perturbation
            adv_images[still_global] = torch.clamp(
                still_imgs.detach() + (1 + self.overshoot) * best_delta,
                min=self.img_scale[0], max=self.img_scale[1]
            ).detach()

        return adv_images.detach()