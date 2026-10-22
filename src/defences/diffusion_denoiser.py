"""
Diffusion Model Denoising Defense for INR-based Classifiers

This module implements defense mechanisms that use diffusion models to
purify adversarial perturbations before classification.

Modes:
- denoise_only: Denoise image in image space
- refit: Denoise image, then refit INR from denoised image (most robust)

References:
- Nie et al., "Diffusion Models for Adversarial Purification" (2022)
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiffusionConfig:
    """Configuration for diffusion denoising defense."""
    timestep: int = 100           # Diffusion timestep for denoising
    multistep: bool = False       # Use multi-step reverse (slower, potentially better)
    mode: str = 'denoise_only'    # 'denoise_only' or 'refit'
    model_path: Optional[str] = None  # Path to diffusion model checkpoint


# Dataset-specific default configurations
# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
DATASET_CONFIGS = {
    'cifar': {
        'image_size': 32,
        # "num_channels": 128,
        # "num_res_blocks": 3,
        # "learn_sigma": True,
        # "dropout": 0.3,
        'channels': 3,
        'model_name': 'cifar10_uncond_50M_500K.pt',
        'normalize_range': (-1, 1),  # Expected input range
    },
    'mnist': {
        'image_size': 28,
        'channels': 1,
        'model_name': 'mnist_diffusion.pt',  # User needs to provide
        'normalize_range': (-1, 1),
    },
    'fashion': {
        'image_size': 28,
        'channels': 1,
        'model_name': 'fashion_diffusion.pt',  # User needs to provide
        'normalize_range': (-1, 1),
    },
}


class DiffusionDenoiser(nn.Module):
    """
    Diffusion-based denoising defense.

    Uses a pre-trained diffusion model to denoise adversarial inputs
    by adding noise at timestep t and then denoising.
    """

    def __init__(
        self,
        dataset: str,
        config: Optional[DiffusionConfig] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            dataset: Dataset name ('cifar', 'mnist', 'fashion')
            config: Diffusion configuration
            device: Device to run on
        """
        super().__init__()
        self.dataset = dataset.lower()
        self.config = config or DiffusionConfig()
        self.device = device

        # Get dataset-specific config
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset}. Supported: {list(DATASET_CONFIGS.keys())}")
        self.dataset_config = DATASET_CONFIGS[self.dataset]

        # Placeholder for diffusion model - will be loaded lazily
        self._diffusion_model = None
        self._diffusion = None

    def _load_diffusion_model(self):
        """Load the diffusion model (lazy loading)."""
        if self._diffusion_model is not None:
            return

        if self.dataset == 'cifar':
            self._load_cifar_diffusion()
        else:
            self._load_generic_diffusion()

    def _load_cifar_diffusion(self):
        """Load CIFAR-10 diffusion model using improved_diffusion."""
        try:
            from .improved_diffusion.script_util import (
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                args_to_dict,
            )

            # CIFAR-10 model args (same as DRM.py)
# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"

            class Args:
                image_size = 32
                num_channels = 128
                num_res_blocks = 3
                num_heads = 4
                num_heads_upsample = -1
                attention_resolutions = "16,8"
                dropout = 0.3
                learn_sigma = True
                sigma_small = False
                class_cond = False
                diffusion_steps = 4000
                noise_schedule = "cosine"
                timestep_respacing = ""
                use_kl = False
                predict_xstart = False
                rescale_timesteps = True
                rescale_learned_sigmas = True
                use_checkpoint = False
                use_scale_shift_norm = True

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(Args(), model_and_diffusion_defaults().keys())
            )

            # Load checkpoint
            model_path = self.config.model_path
            if model_path is None:
                # Default path
                # model_path = "cifar10/cifar10_uncond_50M_500K.pt"
                ckpt_path = "/nfs/stak/users/hongsa/INR-Robustness/defence/cifar/cifar10_uncond_vlb_50M_500K.pt"

            if Path(ckpt_path).exists():
                model.load_state_dict(torch.load(ckpt_path, weights_only=False))
            else:
                print(f"Warning: Diffusion model not found at {ckpt_path}")
                print("Please download the model and set config.model_path")

            model.eval().to(self.device)
            self._diffusion_model = model
            self._diffusion = diffusion

        except ImportError as e:
            raise ImportError(
                f"Failed to import improved_diffusion: {e}. "
                "Make sure improved_diffusion is in the defences directory."
            )

    def _load_generic_diffusion(self):
        """
        Load diffusion model for MNIST/Fashion-MNIST.

        Users should download appropriate models from HuggingFace and
        set the model_path in config.
        """
        model_path = self.config.model_path
        if model_path is None:
            model_path = f"{self.dataset}_diffusion.pt"

        if not Path(model_path).exists():
            print(f"Warning: Diffusion model not found at {model_path}")
            print("Please download a diffusion model for {self.dataset} from HuggingFace")
            print("and set config.model_path")
            print("\nSuggested models:")
            print("  - MNIST: google/ddpm-celebahq-256 (adapt for MNIST)")
            print("  - Fashion: similar unconditional diffusion models")
            self._diffusion_model = None
            self._diffusion = None
            return

        # Try to load from HuggingFace diffusers format
        try:
            from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

            # Check if it's a diffusers checkpoint
            if Path(model_path).is_dir():
                # Load from directory (diffusers format)
                pipeline = DDPMPipeline.from_pretrained(model_path)
                self._diffusion_model = pipeline.unet.to(self.device)
                self._diffusion = pipeline.scheduler
            else:
                # Load from single file (custom format)
                checkpoint = torch.load(model_path, weights_only=False)
                if isinstance(checkpoint, dict):
                    # Assume it's a state dict
                    model = self._create_unet_for_dataset()
                    model.load_state_dict(checkpoint)
                    model.to(self.device)
                    self._diffusion_model = model
                    self._diffusion = DDPMScheduler(num_train_timesteps=1000)
                else:
                    self._diffusion_model = checkpoint.to(self.device)

        except ImportError:
            print("Warning: diffusers library not installed. Install with: pip install diffusers")
            self._diffusion_model = None
            self._diffusion = None

    def _create_unet_for_dataset(self) -> nn.Module:
        """Create a UNet model appropriate for the dataset."""
        try:
            from diffusers import UNet2DModel

            cfg = self.dataset_config
            return UNet2DModel(
                sample_size=cfg['image_size'],
                in_channels=cfg['channels'],
                out_channels=cfg['channels'],
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        except ImportError:
            raise ImportError("diffusers library required for generic diffusion models")

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise input images using diffusion model.

        Args:
            x: Input images, shape (B, C, H, W), range [-1, 1]

        Returns:
            Denoised images, same shape and range
        """
        self._load_diffusion_model()

        if self._diffusion_model is None:
            print("Warning: No diffusion model loaded, returning input unchanged")
            return x

        t = self.config.timestep

        if self.dataset == 'cifar':
            return self._denoise_cifar(x, t)
        else:
            return self._denoise_generic(x, t)

    def _denoise_cifar(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Denoise using CIFAR-10 improved_diffusion model."""
        batch_size = x.shape[0]
        t_batch = torch.tensor([t] * batch_size, device=self.device)

        # Add noise (forward diffusion)
        noise = torch.randn_like(x)
        x_t = self._diffusion.q_sample(x_start=x, t=t_batch, noise=noise)

        # Denoise (reverse diffusion)
        with torch.no_grad():
            if self.config.multistep:
                # Multi-step reverse
                out = x_t
                for i in range(t - 1, -1, -1):
                    t_batch_i = torch.tensor([i] * batch_size, device=self.device)
                    out = self._diffusion.p_sample(
                        self._diffusion_model,
                        out,
                        t_batch_i,
                        clip_denoised=True
                    )['sample']
                return out
            else:
                # Single-step prediction of x_start
                out = self._diffusion.p_sample(
                    self._diffusion_model,
                    x_t,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']
                return out

    def _denoise_generic(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Denoise using generic diffusers model."""
        try:
            from diffusers import DDPMScheduler
        except ImportError:
            return x

        batch_size = x.shape[0]

        # Set timestep
        self._diffusion.set_timesteps(1000)

        # Add noise
        noise = torch.randn_like(x)
        timestep = torch.tensor([t] * batch_size, device=self.device)
        x_t = self._diffusion.add_noise(x, noise, timestep)

        # Denoise
        with torch.no_grad():
            if self.config.multistep:
                # Multi-step reverse
                sample = x_t
                for t_i in range(t, 0, -1):
                    t_tensor = torch.tensor([t_i] * batch_size, device=self.device)
                    model_output = self._diffusion_model(sample, t_tensor).sample
                    sample = self._diffusion.step(model_output, t_i, sample).prev_sample
                return sample
            else:
                # Single step prediction (approximate)
                model_output = self._diffusion_model(x_t, timestep).sample
                # Use predicted noise to recover x_0
                alpha_t = self._diffusion.alphas_cumprod[t]
                x_0_pred = (x_t - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
                return torch.clamp(x_0_pred, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - denoise the input."""
        return self.denoise(x)


class DiffusionDefendedClassifier(nn.Module):
    """
    Classifier with diffusion denoising defense.

    Denoises input images before passing to INR fitting and classification.
    """

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        inr_builder: Optional[Callable] = None,
        classifier: nn.Module = None,
        mode: str = 'denoise_only',
    ):
        """
        Args:
            denoiser: DiffusionDenoiser instance
            inr_builder: Function to fit INR from image (for 'refit' mode)
            classifier: Base classifier (wsfeat -> logits)
            mode: 'denoise_only' or 'refit'
        """
        super().__init__()
        self.denoiser = denoiser
        self.inr_builder = inr_builder
        self.classifier = classifier
        self.mode = mode

        if mode == 'refit' and inr_builder is None:
            raise ValueError("'refit' mode requires inr_builder function")

    def forward(
        self,
        images: torch.Tensor,
        wsfeat=None,
        return_denoised: bool = False,
    ):
        """
        Forward pass with denoising.

        Args:
            images: Input images (B, C, H, W)
            wsfeat: Pre-computed INR weights (ignored in 'refit' mode)
            return_denoised: If True, also return denoised images

        Returns:
            Logits, and optionally denoised images
        """
        # Denoise images
        denoised = self.denoiser(images)

        if self.mode == 'refit':
            # Fit new INR from denoised image
            new_wsfeat = self.inr_builder(denoised)
            logits = self.classifier(new_wsfeat)
        else:
            # Use existing weights with denoised image
            # This mode is less meaningful for INR-based classifiers
            # but included for completeness
            if wsfeat is not None:
                logits = self.classifier(wsfeat)
            else:
                raise ValueError("'denoise_only' mode requires pre-computed wsfeat")

        if return_denoised:
            return logits, denoised
        return logits


class DiffusionRobustModelMultiDataset(nn.Module):
    """
    Extension of DRM.py to support multiple datasets.

    This class combines diffusion denoising with a classifier,
    supporting MNIST, Fashion-MNIST, and CIFAR-10.
    """

    def __init__(
        self,
        dataset: str,
        classifier: Optional[nn.Module] = None,
        config: Optional[DiffusionConfig] = None,
        classifier_type: str = 'vit',  # 'vit' or 'custom'
    ):
        """
        Args:
            dataset: Dataset name
            classifier: Custom classifier (if None, uses default)
            config: Diffusion configuration
            classifier_type: Type of classifier to use
        """
        super().__init__()
        self.dataset = dataset.lower()
        self.config = config or DiffusionConfig()

        # Initialize denoiser
        self.denoiser = DiffusionDenoiser(dataset, config)

        # Initialize classifier
        if classifier is not None:
            self.classifier = classifier
        elif classifier_type == 'vit':
            self.classifier = self._load_vit_classifier()
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def _load_vit_classifier(self) -> nn.Module:
        """Load a ViT classifier for the dataset."""
        try:
            from transformers import AutoModelForImageClassification

            if self.dataset == 'cifar':
                model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
            elif self.dataset == 'mnist':
                model_name = "farleyknight/mnist-digit-classification-2022-09-04"
            elif self.dataset == 'fashion':
                model_name = "abhishek/autotrain_fashion_mnist_vit_base"
            else:
                raise ValueError(f"No default ViT model for dataset: {self.dataset}")

            classifier = AutoModelForImageClassification.from_pretrained(model_name)
            return classifier.eval()

        except ImportError:
            raise ImportError(
                "transformers library required for ViT classifiers. "
                "Install with: pip install transformers"
            )

    def forward(self, x: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with denoising and classification.

        Args:
            x: Input images, range [0, 1]
            t: Diffusion timestep (overrides config if provided)

        Returns:
            Classification logits
        """
        # Convert to [-1, 1] range for diffusion
        x_normalized = x * 2 - 1

        # Override timestep if provided
        if t is not None:
            original_t = self.denoiser.config.timestep
            self.denoiser.config.timestep = t

        # Denoise
        denoised = self.denoiser(x_normalized)

        # Restore timestep
        if t is not None:
            self.denoiser.config.timestep = original_t

        # Resize for ViT if needed
        if hasattr(self.classifier, 'config') and hasattr(self.classifier.config, 'image_size'):
            target_size = self.classifier.config.image_size
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            denoised = torch.nn.functional.interpolate(
                denoised, target_size, mode='bicubic', antialias=True
            )

        # Convert back to [0, 1] for classifier
        denoised_01 = (denoised + 1) / 2

        # Classify
        with torch.no_grad():
            out = self.classifier(denoised_01)

        # Handle different output formats
        if hasattr(out, 'logits'):
            return out.logits
        return out


def create_diffusion_defense(
    dataset: str,
    config: Optional[DiffusionConfig] = None,
    classifier: Optional[nn.Module] = None,
    inr_builder: Optional[Callable] = None,
    mode: str = 'denoise_only',
) -> nn.Module:
    """
    Factory function to create diffusion defense.

    Args:
        dataset: Dataset name
        config: Diffusion configuration
        classifier: Base classifier
        inr_builder: INR fitting function (for 'refit' mode)
        mode: Defense mode

    Returns:
        Defended classifier
    """
    config = config or DiffusionConfig(mode=mode)
    denoiser = DiffusionDenoiser(dataset, config)

    if classifier is None:
        # Return standalone diffusion model
        return DiffusionRobustModelMultiDataset(dataset, config=config)

    return DiffusionDefendedClassifier(
        denoiser=denoiser,
        inr_builder=inr_builder,
        classifier=classifier,
        mode=mode,
    )



def train_diffusion(
    self,
    dataset_name: str,
    base_path: str,
    **kwargs
) -> Dict[str, str]:
    """
    Train a diffusion model for denoising defense.

    Args:
        dataset_name: 'mnist', 'fashion', or 'cifar'
        base_path: Base path for data and checkpoints
        **kwargs: Additional arguments:
            - image_size: Image size (default: 28 for mnist/fashion, 32 for cifar)
            - num_channels: UNet channels (default: 128)
            - num_res_blocks: Number of residual blocks (default: 3)
            - diffusion_steps: Diffusion timesteps (default: 1000)
            - noise_schedule: 'linear' or 'cosine' (default: 'cosine')
            - batch_size: Batch size (default: 128)
            - lr: Learning rate (default: 1e-4)
            - num_iterations: Training iterations (default: 500000)
            - save_interval: Checkpoint save interval (default: 10000)
            - wandb_manager: Optional WandbManager

    Returns:
        Dict with checkpoint path
    """
    import sys
    diffusion_path = os.path.join(self.base_path, "defences", "improved-diffusion")
    if diffusion_path not in sys.path:
        sys.path.insert(0, diffusion_path)

    from improved_diffusion import dist_util, logger
    from improved_diffusion.image_datasets import load_data
    from improved_diffusion.resample import create_named_schedule_sampler
    from improved_diffusion.script_util import create_model_and_diffusion
    from improved_diffusion.train_util import TrainLoop

    # Setup output directory
    path_config = PathConfig(base_path=base_path)
    out_dir = os.path.join(base_path, "defences", "diffusion", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Configure logger
    logger.configure(dir=out_dir)

    # Dataset-specific defaults
    if dataset_name == "cifar":
        default_image_size = 32
        default_channels = 3
    else:  # mnist, fashion
        default_image_size = 28
        default_channels = 1

    # Model and diffusion config
    image_size = kwargs.get('image_size', default_image_size)
    num_channels = kwargs.get('num_channels', 128)
    num_res_blocks = kwargs.get('num_res_blocks', 3)
    diffusion_steps = kwargs.get('diffusion_steps', 1000)
    noise_schedule = kwargs.get('noise_schedule', 'cosine')

    model_kwargs = {
        'image_size': image_size,
        'num_channels': num_channels,
        'num_res_blocks': num_res_blocks,
        'num_heads': kwargs.get('num_heads', 4),
        'num_heads_upsample': kwargs.get('num_heads_upsample', -1),
        'attention_resolutions': kwargs.get('attention_resolutions', '16,8'),
        'dropout': kwargs.get('dropout', 0.0),
        'learn_sigma': kwargs.get('learn_sigma', True),
        'sigma_small': kwargs.get('sigma_small', False),
        'class_cond': kwargs.get('class_cond', False),
        'diffusion_steps': diffusion_steps,
        'noise_schedule': noise_schedule,
        'timestep_respacing': kwargs.get('timestep_respacing', ''),
        'use_kl': kwargs.get('use_kl', False),
        'predict_xstart': kwargs.get('predict_xstart', False),
        'rescale_timesteps': kwargs.get('rescale_timesteps', True),
        'rescale_learned_sigmas': kwargs.get('rescale_learned_sigmas', True),
        'use_checkpoint': kwargs.get('use_checkpoint', False),
        'use_scale_shift_norm': kwargs.get('use_scale_shift_norm', True),
        'in_channels': default_channels,
    }

    print(f"Creating diffusion model for {dataset_name}...")
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        kwargs.get('schedule_sampler', 'uniform'), diffusion
    )

    # Data loader
    data_path = os.path.join(base_path, "raw")
    batch_size = kwargs.get('batch_size', 128)

    print(f"Loading data from {data_path}...")
    data = load_data(
        data_name=dataset_name,
        data_dir=data_path,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=model_kwargs['class_cond'],
    )

    # Training
    print(f"Starting diffusion training...")
    print(f"  Image size: {image_size}")
    print(f"  Channels: {default_channels}")
    print(f"  Diffusion steps: {diffusion_steps}")
    print(f"  Noise schedule: {noise_schedule}")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=kwargs.get('microbatch', -1),
        lr=kwargs.get('lr', 1e-4),
        ema_rate=kwargs.get('ema_rate', '0.9999'),
        log_interval=kwargs.get('log_interval', 100),
        save_interval=kwargs.get('save_interval', 10000),
        resume_checkpoint=kwargs.get('resume_checkpoint', ''),
        use_fp16=kwargs.get('use_fp16', False),
        fp16_scale_growth=kwargs.get('fp16_scale_growth', 1e-3),
        schedule_sampler=schedule_sampler,
        weight_decay=kwargs.get('weight_decay', 0.0),
        lr_anneal_steps=kwargs.get('lr_anneal_steps', 0),
    ).run_loop()

    # Find the latest checkpoint
    checkpoints = glob.glob(os.path.join(out_dir, "model*.pt"))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Diffusion training completed. Checkpoint: {latest_ckpt}")
        return {"checkpoint": latest_ckpt}
    else:
        print("Warning: No checkpoint found after training")
        return {"checkpoint": None}
