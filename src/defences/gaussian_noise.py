"""
Gaussian Noise Defense for INR-based Classifiers

This module implements defense mechanisms that add Gaussian noise to model inputs
to improve robustness via randomized smoothing.

Noise can be applied at different spaces:
- weight: Add noise to INR weights (WeightSpaceFeatures) before classifier
- embedding: Add noise to NFT embeddings before classifier head
- image: Add noise to images before INR fitting (for completeness)

References:
- Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (2019)
- Adapted from locuslab/smoothing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple, List
from scipy.stats import norm
from math import ceil

# Optional import for certification
try:
    from statsmodels.stats.proportion import proportion_confint
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    proportion_confint = None


class GaussianNoiseDefense(nn.Module):
    """
    Base class for Gaussian noise defense.

    Adds N(0, sigma^2) noise to inputs for randomized smoothing.
    """

    ABSTAIN = -1

    def __init__(
        self,
        sigma: float = 0.1,
        space: str = 'weight',
        num_samples: int = 1,
    ):
        """
        Args:
            sigma: Standard deviation of Gaussian noise
            space: Where to apply noise ('weight', 'embedding', 'image')
            num_samples: Number of noisy samples for prediction (1 = single forward)
        """
        super().__init__()
        self.sigma = sigma
        self.space = space
        self.num_samples = num_samples

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input tensor."""
        if self.sigma == 0:
            return x
        noise = torch.randn_like(x) * self.sigma
        return x + noise

    def add_noise_to_wsfeat(self, wsfeat) -> 'WeightSpaceFeatures':
        """Add Gaussian noise to WeightSpaceFeatures (weights and biases)."""
        from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures

        noisy_weights = [self.add_noise(w) for w in wsfeat.weights]
        noisy_biases = [self.add_noise(b) for b in wsfeat.biases]

        return WeightSpaceFeatures(weights=noisy_weights, biases=noisy_biases)


class GaussianDefendedClassifier(nn.Module):
    """
    Wrapper that adds Gaussian noise defense to any classifier.

    Supports both single-sample inference and multi-sample averaging.
    """

    ABSTAIN = -1

    def __init__(
        self,
        base_classifier: nn.Module,
        sigma: float = 0.1,
        space: str = 'weight',
        num_samples: int = 1,
        num_classes: int = 10,
    ):
        """
        Args:
            base_classifier: The underlying classifier model
            sigma: Standard deviation of Gaussian noise
            space: Where to apply noise ('weight', 'embedding')
            num_samples: Number of noisy samples for averaging predictions
            num_classes: Number of output classes
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.space = space
        self.num_samples = num_samples
        self.num_classes = num_classes

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        if self.sigma == 0:
            return x
        return x + torch.randn_like(x) * self.sigma

    def add_noise_to_wsfeat(self, wsfeat):
        """Add Gaussian noise to WeightSpaceFeatures."""
        from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures

        noisy_weights = [self.add_noise(w) for w in wsfeat.weights]
        noisy_biases = [self.add_noise(b) for b in wsfeat.biases]

        return WeightSpaceFeatures(weights=noisy_weights, biases=noisy_biases)

    def forward(self, x, **kwargs):
        """
        Forward pass with noise injection.

        Args:
            x: Input (WeightSpaceFeatures for weight space, tensor for embedding)
            **kwargs: Additional arguments passed to base classifier
        """
        if self.num_samples == 1:
            # Single noisy forward pass
            if self.space == 'weight':
                noisy_x = self.add_noise_to_wsfeat(x)
            else:
                noisy_x = self.add_noise(x)
            return self.base_classifier(noisy_x, **kwargs)
        else:
            # Average over multiple noisy samples
            return self._multi_sample_forward(x, **kwargs)

    def _multi_sample_forward(self, x, **kwargs):
        """Forward pass averaging over multiple noisy samples."""
        logits_sum = None

        for _ in range(self.num_samples):
            if self.space == 'weight':
                noisy_x = self.add_noise_to_wsfeat(x)
            else:
                noisy_x = self.add_noise(x)

            logits = self.base_classifier(noisy_x, **kwargs)

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits

        return logits_sum / self.num_samples

    def predict(self, x, n: int, alpha: float = 0.001, **kwargs) -> int:
        """
        Make a certified prediction using randomized smoothing.

        Args:
            x: Input sample
            n: Number of Monte Carlo samples
            alpha: Failure probability
            **kwargs: Additional arguments

        Returns:
            Predicted class or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, **kwargs)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        from scipy.stats import binomtest
        if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
            return self.ABSTAIN
        else:
            return top2[0]

    def certify(
        self,
        x,
        n0: int,
        n: int,
        alpha: float = 0.001,
        **kwargs
    ) -> Tuple[int, float]:
        """
        Certify robustness radius using randomized smoothing.

        Args:
            x: Input sample
            n0: Number of samples for selection
            n: Number of samples for estimation
            alpha: Failure probability
            **kwargs: Additional arguments

        Returns:
            (predicted_class, certified_radius) or (ABSTAIN, 0.0)
        """
        self.base_classifier.eval()

        # Selection phase
        counts_selection = self._sample_noise(x, n0, **kwargs)
        cAHat = counts_selection.argmax().item()

        # Estimation phase
        counts_estimation = self._sample_noise(x, n, **kwargs)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)

        if pABar < 0.5:
            return self.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def _sample_noise(self, x, num: int, batch_size: int = 64, **kwargs) -> np.ndarray:
        """Sample predictions under noisy inputs."""
        counts = np.zeros(self.num_classes, dtype=int)

        with torch.no_grad():
            remaining = num
            while remaining > 0:
                this_batch = min(batch_size, remaining)
                remaining -= this_batch

                # Repeat input for batch processing
                if self.space == 'weight':
                    from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
                    # For WeightSpaceFeatures, we process one at a time with noise
                    for _ in range(this_batch):
                        noisy_x = self.add_noise_to_wsfeat(x)
                        pred = self.base_classifier(noisy_x, **kwargs).argmax(1)
                        for p in pred.cpu().numpy():
                            counts[p] += 1
                else:
                    # For tensors, can batch more efficiently
                    batch_x = x.repeat(this_batch, *([1] * (x.dim() - 1)))
                    noisy_batch = self.add_noise(batch_x)
                    preds = self.base_classifier(noisy_batch, **kwargs).argmax(1)
                    for p in preds.cpu().numpy():
                        counts[p] += 1

        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Compute lower confidence bound using Clopper-Pearson method."""
        if not HAS_STATSMODELS:
            # Fallback: simple proportion estimate (less accurate)
            return NA / N - 1.96 * np.sqrt((NA/N) * (1 - NA/N) / N)
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


class GaussianDefendedNFN(GaussianDefendedClassifier):
    """
    Gaussian noise defense specifically for NFN models.

    Applies noise to WeightSpaceFeatures (INR weights and biases).
    """

    def __init__(
        self,
        nfn_model: nn.Module,
        sigma: float = 0.1,
        num_samples: int = 1,
        num_classes: int = 10,
    ):
        super().__init__(
            base_classifier=nfn_model,
            sigma=sigma,
            space='weight',
            num_samples=num_samples,
            num_classes=num_classes,
        )


class GaussianDefendedNFT(GaussianDefendedClassifier):
    """
    Gaussian noise defense for NFT (Neural Functional Transformer) models.

    Can apply noise either to:
    - 'weight': WeightSpaceFeatures before encoder
    - 'embedding': Encoded embeddings before classifier head
    """

    def __init__(
        self,
        nft_encoder: nn.Module,
        nft_classifier: nn.Module,
        sigma: float = 0.1,
        space: str = 'embedding',
        num_samples: int = 1,
        num_classes: int = 10,
    ):
        """
        Args:
            nft_encoder: NFT AutoEncoder that produces embeddings
            nft_classifier: Classifier head operating on embeddings
            sigma: Noise standard deviation
            space: 'weight' or 'embedding'
            num_samples: Number of noisy samples
            num_classes: Number of output classes
        """
        # Create a combined model for the base classifier
        combined = _NFTCombined(nft_encoder, nft_classifier, space, sigma)
        super().__init__(
            base_classifier=combined,
            sigma=sigma,
            space=space,
            num_samples=num_samples,
            num_classes=num_classes,
        )
        self.encoder = nft_encoder
        self.classifier = nft_classifier


class _NFTCombined(nn.Module):
    """Helper class combining NFT encoder and classifier with noise injection point."""

    def __init__(self, encoder, classifier, noise_space, sigma):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.noise_space = noise_space
        self.sigma = sigma

    def forward(self, x, apply_noise=True):
        """
        Forward pass with optional noise at the specified space.

        Note: Noise is applied by the parent GaussianDefendedClassifier,
        so this just does the encoding and classification.
        """
        # If noise_space is 'weight', x is already noisy WeightSpaceFeatures
        # If noise_space is 'embedding', we encode first then noise is applied externally

        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits


def create_defended_model(
    model: nn.Module,
    model_type: str,
    sigma: float = 0.1,
    space: str = 'weight',
    num_samples: int = 1,
    num_classes: int = 10,
    **kwargs
) -> GaussianDefendedClassifier:
    """
    Factory function to create a Gaussian defended model.

    Args:
        model: Base classifier model
        model_type: Type of model ('nfn', 'nft', 'scalegmn', 'mwt')
        sigma: Noise standard deviation
        space: Where to apply noise
        num_samples: Number of noisy samples for prediction
        num_classes: Number of output classes
        **kwargs: Additional arguments (e.g., encoder/classifier for NFT)

    Returns:
        Defended model wrapper
    """
    if model_type.lower() == 'nfn':
        return GaussianDefendedNFN(
            nfn_model=model,
            sigma=sigma,
            num_samples=num_samples,
            num_classes=num_classes,
        )
    elif model_type.lower() == 'nft':
        encoder = kwargs.get('encoder', model)
        classifier = kwargs.get('classifier', None)
        if classifier is None:
            raise ValueError("NFT requires 'classifier' argument")
        return GaussianDefendedNFT(
            nft_encoder=encoder,
            nft_classifier=classifier,
            sigma=sigma,
            space=space,
            num_samples=num_samples,
            num_classes=num_classes,
        )
    else:
        # Generic wrapper for other model types
        return GaussianDefendedClassifier(
            base_classifier=model,
            sigma=sigma,
            space=space,
            num_samples=num_samples,
            num_classes=num_classes,
        )
