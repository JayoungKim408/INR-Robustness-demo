import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union
from dataclasses import dataclass
from attacks.torchattacks.attacks.pgd import PGD
from attacks.torchattacks.attacks.pgdwsfeat import PGDWSFeat
from attacks.torchattacks.attacks.pgdembedding import PGDEmbedding, compute_embedding_stats
from attacks.utils import DEFAULT_EPS, DEFAULT_EPS_STD
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures


@dataclass
class ATConfig:
    eps: float = 0.1           # Perturbation bound
    eps_img: float = 0.3
    steps: int = 10            # PGD steps
    alpha: float = 0.01        # Step size (default: 2*eps/steps if None)
    lambda_adv: float = 0.5    # Weight for adversarial loss
    random_start: bool = True  # Random initialization within eps ball

class AdversarialTraining(nn.Module):
    """Unified adversarial training for both INR weight space and embedding space."""

    def __init__(
        self,
        victim_wrapper: nn.Module,
        config: Optional[ATConfig] = None,
        mode: str = "inr",  # "inr" or "embedding"
        renderer: Callable = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.victim_wrapper = victim_wrapper
        self.config = config or ATConfig()
        self.mode = mode
        self.device = device

        if mode == "embedding_at":
            self.classifier = victim_wrapper.clf
            self.pgd = PGDEmbedding(
                model=self.classifier,
                eps_std=self.config.eps,
                alpha_std=self.config.alpha,
                steps=self.config.steps,
            )
        elif mode == "inr_at":
            self.pgd = PGDWSFeat(
                model=victim_wrapper,
                eps=self.config.eps * 2,
                eps_img=self.config.eps * 2,
                steps=self.config.steps,
                proj_mode="icop",
                renderer=renderer,
            )
        else:
            raise ValueError(f"Unknown AT mode: {mode}")

    def forward(self, inputs, labels: torch.Tensor):
        if self.mode == "embedding_at":
            with torch.no_grad():
                clean_emb = self.victim_wrapper.encode(inputs)
            adv_emb = self.pgd(clean_emb.detach().clone(), labels)
            clean_logits = self.classifier(clean_emb)
            adv_logits = self.classifier(adv_emb)
        else:
            clean_logits = self.victim_wrapper(inputs)
            adv_inputs = self.pgd(inputs, labels)
            adv_logits = self.victim_wrapper(adv_inputs)

        return clean_logits, adv_logits

    def train_step(self, inputs, labels: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict:
        optimizer.zero_grad()
        clean_logits, adv_logits = self.forward(inputs, labels)

        clean_loss = F.cross_entropy(clean_logits, labels)
        adv_loss = F.cross_entropy(adv_logits, labels)

        lam = self.config.lambda_adv
        total_loss = (1 - lam) * clean_loss + lam * adv_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_acc = (clean_logits.argmax(1) == labels).float().mean().item()
            adv_acc = (adv_logits.argmax(1) == labels).float().mean().item()

        return {
            'total_loss': total_loss.item(),
            'clean_loss': clean_loss.item(),
            'adv_loss': adv_loss.item(),
            'clean_acc': clean_acc,
            'adv_acc': adv_acc,
        }

    def validation(self, dataloader) -> dict:
        self.victim_wrapper.eval()
        total_loss, total_clean_loss, total_adv_loss = 0, 0, 0
        total_clean_acc, total_adv_acc = 0, 0

        for batch in dataloader:
            params, init_params, img, label = batch
            wsfeat = WeightSpaceFeatures(*params).to(self.device)
            label = label.to(self.device)

            clean_logits, adv_logits = self.forward(wsfeat, label)

            clean_loss = F.cross_entropy(clean_logits, label)
            adv_loss = F.cross_entropy(adv_logits, label)

            lam = self.config.lambda_adv
            loss = (1 - lam) * clean_loss + lam * adv_loss

            with torch.no_grad():
                clean_acc = (clean_logits.argmax(1) == label).float().mean().item()
                adv_acc = (adv_logits.argmax(1) == label).float().mean().item()

            total_loss += loss.item()
            total_clean_loss += clean_loss.item()
            total_adv_loss += adv_loss.item()
            total_clean_acc += clean_acc
            total_adv_acc += adv_acc
        self.victim_wrapper.train()

        return {
            'val_loss': total_loss / len(dataloader),
            'val_clean_loss': total_clean_loss / len(dataloader),
            'val_adv_loss': total_adv_loss / len(dataloader),
            'val_clean_acc': total_clean_acc / len(dataloader),
            'val_adv_acc': total_adv_acc / len(dataloader),
        }
