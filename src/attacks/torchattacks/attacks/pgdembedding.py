import torch
import torch.nn as nn

from ..attack import Attack
from attacks.utils import assert_within_eps


class PGDEmbedding(Attack):
    r"""
    PGD Attack for Embedding Space with std-normalized epsilon.
    
    Perturbation is bounded relative to embedding statistics:
        ||δ||∞ ≤ eps_std * std(embeddings)
    
    This allows fair comparison with image-space attacks when eps_std
    is set to match the image epsilon in std units.

    Arguments:
        model (nn.Module): classifier model that takes embeddings as input.
        eps_std (float): maximum perturbation in std units. (Default: 0.1)
            - CIFAR-equivalent: ~0.06
            - Fashion-equivalent: ~0.14  
            - MNIST-equivalent: ~0.48
        alpha_std (float): step size in std units. (Default: eps_std/4)
        steps (int): number of steps. (Default: 10)
        emb_std (torch.Tensor or float): pre-computed embedding std.
            - If None, computed from first batch (not recommended).
            - Can be scalar (global std) or per-dimension tensor.
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - embeddings: :math:`(N, *)` where `*` is any shape (e.g., (N, 16, 256) for NFT).
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, *)` same shape as input embeddings.

    Examples::
        >>> # Pre-compute embedding std from dataset
        >>> all_embs = torch.cat([emb for emb, _ in dataloader])
        >>> emb_std = all_embs.std()
        >>> 
        >>> # Attack with MNIST-equivalent strength (0.48 std)
        >>> attack = PGDEmbedding(model, eps_std=0.48, emb_std=emb_std)
        >>> adv_embs = attack(embeddings, labels)
    """

    def __init__(
        self,
        model=None,
        eps_std=0.1,
        alpha_std=None,
        steps=20,
        emb_std=None,
        random_start=True,
    ):
        super().__init__("PGDEmbedding", model)
        self.eps_std = eps_std
        self.alpha_std = alpha_std if alpha_std is not None else eps_std / 4
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        
        # Store embedding std (can be set later via set_emb_std)
        self._emb_std = None
        if emb_std is not None:
            self.set_emb_std(emb_std)

    def set_emb_std(self, emb_std):
        """Set pre-computed embedding std."""
        if isinstance(emb_std, (int, float)):
            self._emb_std = torch.tensor(emb_std, dtype=torch.float32)
        else:
            self._emb_std = emb_std.clone().detach()
    
    @property
    def eps(self):
        """Actual epsilon in embedding units."""
        if self._emb_std is None:
            raise ValueError("emb_std not set. Call set_emb_std() or pass emb_std to __init__.")
        return self.eps_std * self._emb_std
    
    @property
    def alpha(self):
        """Actual step size in embedding units."""
        if self._emb_std is None:
            raise ValueError("emb_std not set. Call set_emb_std() or pass emb_std to __init__.")
        return self.alpha_std * self._emb_std

    def forward(self, embeddings, labels, model=None):
        r"""
        Overridden.
        """
        if model is not None: # lazy set model
            self.set_model(model)
            self.set_device(next(model.parameters()).device)

        embeddings = embeddings.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(embeddings, labels)

        # Ensure emb_std is set and on correct device
        if self._emb_std is None:
            # Fallback: compute from batch (not recommended)
            print("Warning: Computing emb_std from batch. Pre-compute from full dataset for stability.")
            self._emb_std = embeddings.std()
        
        emb_std = self._emb_std.to(self.device)
        eps = self.eps_std * emb_std
        alpha = self.alpha_std * emb_std

        loss = nn.CrossEntropyLoss()
        adv_embeddings = embeddings.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point within eps-ball
            adv_embeddings = adv_embeddings + torch.empty_like(adv_embeddings).uniform_(-1, 1) * eps
            # No clamping to valid range (embeddings are unbounded)

        for _ in range(self.steps):
            adv_embeddings.requires_grad = True
            outputs = self.get_logits(adv_embeddings)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial embeddings
            grad = torch.autograd.grad(
                cost, adv_embeddings, retain_graph=False, create_graph=False
            )[0]

            adv_embeddings = adv_embeddings.detach() + alpha * grad.sign()
            
            # Project back to eps-ball (L-inf)
            delta = torch.clamp(adv_embeddings - embeddings, min=-eps, max=eps)
            adv_embeddings = embeddings + delta
            # No clamping to [0,1] - embeddings are unbounded

        pass # need to check! 
        self.check_projection(embeddings, adv_embeddings)
        return adv_embeddings

    def check_projection(self, imgs, adv_imgs):
        report, stats, arrays = assert_within_eps(imgs, adv_imgs, self.eps, norm='linf')

    
# Utility function to compute embedding statistics
def compute_embedding_stats(dataloader):
    """
    Compute embedding statistics from a dataloader.
    
    Args:
        dataloader: DataLoader yielding (embeddings, labels) or (images, labels)
        encoder: If provided, encode images to embeddings. If None, assume dataloader yields embeddings.
        device: Device to use.
    
    Returns:
        dict with 'mean', 'std', 'min', 'max' of embeddings
    """
    all_embs = []
    
    with torch.no_grad():
        for _, _, _, embs, _ in dataloader:
            all_embs.append(embs.cpu())
    all_embs = torch.cat(all_embs, dim=0)
    
    return {
        'mean': all_embs.mean(),
        'std': all_embs.std(),
        'std_per_dim': all_embs.std(dim=0),
        'min': all_embs.min(),
        'max': all_embs.max(),
    }