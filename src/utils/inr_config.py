import os
from dataclasses import dataclass


@dataclass
class INRConfig:
    """Configuration for SIREN/FFN INR generation and lookup.

    Attributes:
        depth: Number of hidden layers (0, 1, 3)
        width: Number of hidden features per layer (16, 32, 64)
        scale: Frequency scaling factor first_omega_0 (10, 30, 60)
        extra_aug: Number of augmented INRs per image (1, 5, 10, 20)
        training_steps: Number of optimization steps (default 5000)
    """
    depth: int = 1
    width: int = 32
    scale: int = 30
    extra_aug: bool = False
    training_steps: int = 5000
    prefix: str = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.depth not in [0, 1, 3, 4]:
            raise ValueError(f"depth must be 0, 1, or 3, got {self.depth}")
        if self.width not in [16, 32, 64, 128, 512]:
            raise ValueError(f"width must be 16, 32, or 64, got {self.width}")
        if self.scale not in [10, 30, 60]:
            raise ValueError(f"scale must be 10, 30, or 60, got {self.scale}")
        if self.extra_aug not in [True, False]:
            raise ValueError(f"extra_aug must be True or False, got {self.extra_aug}")

    def get_inr_path(self, base_path: str, dataset: str) -> str:
        config_dir = f"depth{self.depth}_width{self.width}_scale{self.scale}"
        return os.path.join(base_path, "assets", "sirens", config_dir, dataset)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "depth": self.depth,
            "width": self.width,
            "scale": self.scale,
            "extra_aug": self.extra_aug,
            "training_steps": self.training_steps,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "INRConfig":
        """Create INRConfig from dictionary."""
        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation."""
        return f"INR(d={self.depth}, w={self.width}, s={self.scale}, aug={self.extra_aug}, prefix={self.prefix})"
