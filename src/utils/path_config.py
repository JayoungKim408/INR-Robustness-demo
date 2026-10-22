"""
Centralized path configuration for INR-Robustness project.

Provides standardized paths for:
- Model checkpoints (attacks, basemodels)
- Training outputs (plots, logs)
- Data (siren weights, raw data)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


# Default base path - can be overridden via environment variable
DEFAULT_BASE_PATH = os.environ.get("INR_ROBUSTNESS_BASE_PATH", None)


@dataclass
class PathConfig:

    base_path: str = field(default_factory=lambda: DEFAULT_BASE_PATH)

    def __post_init__(self):
        """Validate and normalize base path."""
        if self.base_path is None:
            raise ValueError(
                "No base_path provided for PathConfig. "
                "Set the INR_ROBUSTNESS_BASE_PATH environment variable or pass base_path explicitly."
            )
        self.base_path = os.path.expanduser(self.base_path)
        self.base_path = os.path.abspath(self.base_path)
        self.data_path = os.path.join(self.base_path, "assets", "raw")

    # =========================================================================
    # Surrogate Checkpoint Paths
    # =========================================================================

    def get_surrogate_checkpoint_dir(
        self,
        surrogate_type: str,
        dataset: str,
        victim_model: Optional[str] = None,
        inr_params: Optional[Any] = None,
        latent_dim: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Get checkpoint directory for a specific surrogate type.

        Args:
            surrogate_type: Surrogate type (i2i-fd, i2i-naive, i2e-fd, i2e-naive,
                        i2i-hypernet, gaussian, transfer, e2e)
            dataset: Dataset name (mnist, fashion, cifar)
            victim_model: Victim model name (nft, nfn, scalegmn, dws, mwt)
            **kwargs: Additional parameters for path generation (e.g., lr, lam_emb)

        Returns:
            Path to checkpoint directory
        """
        # surrogate_type_lower = surrogate_type.lower()
        inr_name = f"depth{inr_params.depth}_width{inr_params.width}_scale{inr_params.scale}"
        
        if surrogate_type in ['transfer']:
            path = os.path.join(
                self.base_path, "assets", "attacks", surrogate_type
            )
        else: 
            path = os.path.join(
                self.base_path, "assets", "attacks", surrogate_type, dataset, victim_model
            )

        if inr_name != "depth1_width32_scale30": # base
            path = os.path.join(path, inr_name)

        if latent_dim != 256 and surrogate_type.lower() not in ("gaussian", "e2e"):
            path = os.path.join(path, f"latent_dim_{latent_dim}")

        return path

    def get_surrogate_checkpoint_path(
        self,
        surrogate_type: str,
        dataset: str,
        victim_model: str,
        attack_params: Dict[str, Any],
        inr_params: Optional[Dict[str, Any]] = None,
        # latent_dim: Optional[int] = None,
    ) -> str:
        """
        Generate standardized checkpoint path for an attack.

        Uses PathConfig for consistent path generation across the project.

        Args:
            surrogate_type: Surrogate type (i2i-fd, i2e-naive, etc.)
            dataset: Dataset name
            victim_model: Victim model name
            attack_params: Attack parameters (used for param-based subdirs)

        Returns:
            Path to checkpoint directory
        """
        # Build param string for subdirectory naming
        param_parts = []

        if surrogate_type == "i2i-hypernet":
            lr = attack_params.get("lr", 5e-5)
            kl_weight = attack_params.get("kl_weight", 0.1)
            fw_weight = attack_params.get("fw_weight", 100)
            cls_weight = attack_params.get("cls_weight", 1.0)
            param_parts = [str(lr), str(kl_weight), str(fw_weight), str(cls_weight)]

        # surrogates 
        elif surrogate_type in ("i2i-fd", "i2e-fd"):
            lr = attack_params.get("lr", 5e-5)
            lam_emb = attack_params.get("lam_emb", 1.0)
            lam_inr = attack_params.get("lam_inr", 100.0)
            lam_der = attack_params.get("lam_der", 0.1 if surrogate_type == 'i2i-fd' else 0.001)
            lam_cls = attack_params.get("lam_cls", 0.1)
            lam_recon = attack_params.get("lam_recon", 0.1)
            alpha = attack_params.get("alpha", 0.0001)
            
            # augmentation
            n_aug = attack_params.get("n_aug", 0)
            pgd_aug = attack_params.get("pgd_aug", 0)
            norm = attack_params.get("norm", "linf")
            if pgd_aug:
                surr_train_steps = attack_params.get("surr_train_steps", 5)
                aug_type = f"pgd_{norm}_steps_{surr_train_steps}"
            elif n_aug > 0:
                aug_type = f"random_{n_aug}"
            else:
                aug_type = ""
            # aug_eps = attack_params.get("aug_eps", 0.0001)
            param_parts = [str(lr), 
                           str(lam_emb) if surrogate_type == 'i2e-fd' else str(lam_inr), 
                           str(lam_der), 
                           str(lam_cls) if surrogate_type == 'i2i-fd' else "", 
                           str(lam_recon), 
                           str(alpha), 
                           aug_type]

        elif surrogate_type in ("i2i-naive", "i2e-naive"):
            lr = attack_params.get("lr", 5e-5)
            lam_inr = attack_params.get("lam_inr", 100.0)
            lam_cls = attack_params.get("lam_cls", 0.1)
            lam_recon = attack_params.get("lam_recon", 0.1)

            param_parts = [str(lr)] if surrogate_type == "i2e-naive" else [str(lr), str(lam_inr), str(lam_cls), str(lam_recon)]

        # Get base checkpoint directory from PathConfig
        ckpt_dir = self.get_surrogate_checkpoint_dir(
            surrogate_type=surrogate_type,
            dataset=dataset,
            victim_model=victim_model,
            inr_params=inr_params,
            latent_dim=attack_params.get("latent_dim", 256),
        )

        # Append param-based subdirectory if we have params
        if param_parts:
            param_str = "_".join(param_parts)
            ckpt_dir = os.path.join(ckpt_dir, f"{dataset}_{param_str}")

        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    # =========================================================================
    # Attack Plot/Output Paths
    # =========================================================================

    def get_attack_plot_dir(
        self,
        attack_type: str,
        dataset: str,
        surrogate_type: Optional[str] = None,
        victim_model: Optional[str] = None,
    ) -> str:
        
        if surrogate_type is not None:
            return os.path.join(
                self.base_path, "plots", attack_type, surrogate_type.lower().replace("-", "_") , victim_model, dataset
            )
        else:
            return os.path.join(
                self.base_path, "plots", attack_type, victim_model, dataset
            )

    # =========================================================================
    # Base Model Paths
    # =========================================================================

    def get_basemodel_dir(
        self,
        model: str,
        dataset: str,
        inr_config: Optional[Any] = None,
        latent_dim: Optional[int] = None,
        at_type: Optional[str] = None,
    ) -> str:
        if at_type is not None:
            base_dir = os.path.join(self.base_path, "assets", "defenses", model, dataset)
        else:
            base_dir = os.path.join(self.base_path, "assets", "basemodels", model, dataset)

        if inr_config is not None:
            inr_subdir = self._get_inr_config_subdir(inr_config)
            if inr_subdir:
                base_dir = os.path.join(base_dir, inr_subdir)

        if latent_dim is not None and latent_dim != 256:
            base_dir = os.path.join(base_dir, f"latent_dim_{latent_dim}")

        return base_dir

    def _get_inr_config_subdir(self, inr_config: Any) -> Optional[str]:
        return f"inr_depth{inr_config.depth}_width{inr_config.width}_scale{int(inr_config.scale)}"

    def get_basemodel_checkpoint(
        self,
        model: str,
        dataset: str,
        checkpoint_name: str = "best.pt",
        inr_config: Optional[Any] = None,
        latent_dim: Optional[int] = None,
        defense_type: Optional[str] = None,
        defense_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        defense_dir = None
        if defense_type is not None:
            defense_dir = self.get_basemodel_dir(model, dataset, inr_config, latent_dim, at_type=defense_type)
        basemodel_dir = self.get_basemodel_dir(model, dataset, inr_config, latent_dim)

        if model.split("_")[0] in ["nft", "inr2vec"]:
            inr2array_dir = os.path.join(basemodel_dir, "encoder", checkpoint_name)
            classifier_dir = os.path.join(basemodel_dir if defense_dir is None else defense_dir, "classifier", checkpoint_name)
            return {"basemodel": inr2array_dir, "classifier": classifier_dir}
        else:
            return {"basemodel": os.path.join(basemodel_dir if defense_dir is None else defense_dir, checkpoint_name)}

    def get_embedding_path(
        self,
        model: str,
        dataset: str,
        inr_config: Optional[Any] = None,
        latent_dim: Optional[int] = None,
    ) -> str:
        if model.split("_")[0] in ["nft", "inr2vec"]:
            dir = os.path.join(self.get_basemodel_dir(model, dataset, inr_config, latent_dim), "embeddings")
            return dir if dataset == "shapenet" else os.path.join(dir, "embeddings.pt")
        else: 
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        return f"PathConfig(base_path='{self.base_path}')"
