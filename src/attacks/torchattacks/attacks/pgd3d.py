import torch
import torch.nn as nn
from ..attack import Attack

class PGD3D(Attack):
    def __init__(self, 
                 model=None,
                 eps=0.05,           # ← 변경: 8/255 → 0.05
                 alpha=0.005,        # ← 변경: 2/255 → 0.005
                 steps=200,          # ← 변경: 10 → 200
                 point_scale=[-1, 1],  # ← 변경: [0,1] → [-1,1]
                 random_start=True
                 ):
        super().__init__("PGD3D", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.point_scale = point_scale
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, points, labels, model=None):
        # points: (B, N, 3) — unit-sphere normalized
        if model is not None:
            self.set_model(model)
            self.set_device(next(model.parameters()).device)

        points = points.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(points, labels)

        loss = nn.CrossEntropyLoss()
        adv_points = points.clone().detach()

        if self.random_start:
            adv_points = adv_points + torch.empty_like(adv_points).uniform_(
                -self.eps, self.eps
            )
            adv_points = torch.clamp(adv_points, 
                                     min=self.point_scale[0], 
                                     max=self.point_scale[1]).detach()

        for _ in range(self.steps):
            adv_points.requires_grad = True
            outputs = self.get_logits(adv_points)  # ★ 아래 참고

            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_points, 
                                       retain_graph=False, create_graph=False)[0]

            adv_points = adv_points.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_points - points, min=-self.eps, max=self.eps)
            adv_points = torch.clamp(points + delta, 
                                     min=self.point_scale[0], 
                                     max=self.point_scale[1]).detach()
            
        self.check_projection(points, adv_points)
        return adv_points