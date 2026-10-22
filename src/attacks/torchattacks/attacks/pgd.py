import torch
import torch.nn as nn

from ..attack import Attack
from attacks.utils import assert_within_eps


class PGD(Attack):
    
    def __init__(self, 
                 model=None,
                 eps=8 / 255, 
                 alpha=2 / 255, 
                 steps=10, 
                 img_scale=[0, 1], 
                 random_start=True
                 ):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.img_scale = img_scale
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

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

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=self.img_scale[0], max=self.img_scale[1]).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.img_scale[0], max=self.img_scale[1]).detach()
            
        self.check_projection(images, adv_images)
        return adv_images

    def check_projection(self, imgs, adv_imgs):
        report, stats, arrays = assert_within_eps(imgs, adv_imgs, self.eps, norm='linf')
        # print("projection check:", report)
