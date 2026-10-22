"""
Centralized wandb management for INR-Robustness experiments.

Provides:
- Unified wandb.init/finish lifecycle
- Consistent naming conventions
- Support for standalone and suite-based execution
- Graceful disabled mode (no-op when wandb is disabled)

Usage:
    # Standalone mode with context manager
    manager = WandbManager(enabled=True)
    with manager.run(name="my_run", config=my_config):
        manager.log({"loss": 0.5})

    # Suite mode (managed externally)
    manager = WandbManager(enabled=True, suite_name="exp1")
    manager.init_run(name="mnist_nft_gaussian", group="exp1", ...)
    # ... experiment code ...
    manager.finish_run()
"""

import os
import wandb
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager


@dataclass
class WandbConfig:
    """Configuration for wandb run."""
    enabled: bool = False
    project: str = "INR-Robustness-exp"

    # Run identification
    name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

    # Resume support
    run_id: Optional[str] = None
    resume: Optional[str] = None  # "allow", "must", "never", or None

    # Logging settings
    log_freq: int = 10
    save_code: bool = False


class WandbManager:
    """
    Centralized wandb run manager for INR-Robustness.

    Supports two modes:
    1. Standalone mode: Direct script execution (e.g., python train_surrogate_inr.py --wandb)
    2. Suite mode: Orchestrated by ExperimentSuite with managed run lifecycle

    When disabled, all methods become no-ops, allowing code to run without wandb
    without needing conditional checks everywhere.

    Attributes:
        enabled: Whether wandb logging is active
        suite_name: Optional suite name for grouping experiments
        project: Wandb project name
        current_run: Active wandb.Run object or None
    """

    DEFAULT_PROJECT = "INR-Robustness-exp"

    def __init__(
        self,
        enabled: bool = False,
        suite_name: Optional[str] = None,
        project: Optional[str] = None,
        config: Optional[WandbConfig] = None,
    ):
        """Initialize WandbManager.

        Args:
            enabled: Whether wandb is enabled
            suite_name: Optional suite name for grouping experiments
            project: Wandb project name (defaults to "INR-Robustness-exp")
            config: Optional WandbConfig override
        """
        self.enabled = enabled
        self.suite_name = suite_name
        self.project = project or self.DEFAULT_PROJECT
        self.config = config or WandbConfig(enabled=enabled, project=self.project)
        self.current_run: Optional[wandb.sdk.wandb_run.Run] = None
        self._is_suite_mode = suite_name is not None

    @classmethod
    def from_args(cls, args, suite_name: Optional[str] = None) -> "WandbManager":
        """Create WandbManager from argparse namespace.

        Args:
            args: argparse.Namespace with optional 'wandb' attribute
            suite_name: Optional suite name

        Returns:
            Configured WandbManager
        """
        enabled = getattr(args, 'wandb', False)
        return cls(enabled=enabled, suite_name=suite_name)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], suite_name: Optional[str] = None) -> "WandbManager":
        """Create WandbManager from OmegaConf or dict config.

        Args:
            cfg: Configuration dict with optional 'wandb' key
            suite_name: Optional suite name

        Returns:
            Configured WandbManager
        """
        # Handle both nested and flat config structures
        if isinstance(cfg, dict):
            if 'wandb' in cfg:
                if isinstance(cfg['wandb'], dict):
                    enabled = cfg['wandb'].get('enabled', False)
                else:
                    enabled = bool(cfg['wandb'])
            else:
                enabled = False
        else:
            # Handle OmegaConf or similar
            try:
                if hasattr(cfg, 'wandb'):
                    wandb_cfg = cfg.wandb
                    if hasattr(wandb_cfg, 'enabled'):
                        enabled = wandb_cfg.enabled
                    else:
                        enabled = bool(wandb_cfg)
                else:
                    enabled = False
            except Exception:
                enabled = False

        return cls(enabled=enabled, suite_name=suite_name)

    def generate_run_name(
        self,
        dataset: str,
        model: str,
        attack_type: Optional[str] = None,
        surrogate_type: Optional[str] = None,
        training_type: Optional[str] = None,
        suffix: Optional[str] = None,
        include_timestamp: bool = True,
    ) -> str:
        """Generate standardized run name.

        Format for attacks: "{dataset}_{model}_{attack_type}_{timestamp}"
        Format for training: "{training_type}_{dataset}_{model}_{timestamp}"

        Args:
            dataset: Dataset name (mnist, fashion, cifar)
            model: Model name (nft, nfn, scalegmn, mwt)
            attack_type: Attack type for attack runs
            training_type: Training type for training runs (e.g., "surrogate", "inr2array")
            suffix: Optional suffix for disambiguation
            include_timestamp: Whether to include timestamp in name

        Returns:
            Formatted run name
        """
        if training_type:
            name = f"{training_type}_{dataset}_{model}"
        elif attack_type:
            if surrogate_type:
                name = f"{dataset}_{model}_{attack_type}_{surrogate_type}"
            else:
                name = f"{dataset}_{model}_{attack_type}"
        else:
            name = f"{dataset}_{model}"

        if suffix:
            name = f"{name}_{suffix}"

        if include_timestamp:
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            name = f"{name}_{timestamp}"

        return name

    def generate_group_name(
        self,
        experiment_type: str,
        dataset: str,
        attack_type: Optional[str] = None,
        surrogate_type: Optional[str] = None,
    ) -> str:
        """Generate standardized group name.

        Format: "{suite_name}" if in suite mode
                "{experiment_type}_{attack_type}_{dataset}" otherwise

        Args:
            experiment_type: Type of experiment (e.g., "attack", "training")
            dataset: Dataset name
            attack_type: Optional attack type for more specific grouping
            surrogate_type: Optional surrogate type for more specific grouping
        Returns:
            Group name string
        """
        if self.suite_name:
            return self.suite_name

        if attack_type:
            if surrogate_type:
                return f"{experiment_type}_{attack_type}_{surrogate_type}_{dataset}"
            return f"{experiment_type}_{attack_type}_{dataset}"
        return f"{experiment_type}_{dataset}"

    def generate_tags(
        self,
        dataset: str,
        model: str,
        attack_space: Optional[str] = None,
        attack_type: Optional[str] = None,
        surrogate_type: Optional[str] = None,
        training_type: Optional[str] = None,
        extra_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate standardized tags for run.

        Args:
            dataset: Dataset name
            model: Model name
            attack_space: Attack space (image, inr, embedding)
            attack_type: Attack type
            training_type: Training type (for training runs)
            extra_tags: Additional custom tags

        Returns:
            List of tag strings
        """
        tags = [dataset, model]

        if attack_space:
            tags.append(f"space:{attack_space}")
        if attack_type:
            tags.append(f"attack:{attack_type}")
        if surrogate_type:
            tags.append(f"surrogate:{surrogate_type}")
        if training_type:
            tags.append(f"training:{training_type}")
        if self.suite_name:
            tags.append(f"suite:{self.suite_name}")

        if extra_tags:
            tags.extend(extra_tags)

        return tags

    def init_run(
        self,
        name: str,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: Optional[str] = None,
        reinit: bool = True,
        job_type: Optional[str] = None,
    ) -> Optional[wandb.sdk.wandb_run.Run]:
        """Initialize a new wandb run.

        Args:
            name: Run name
            group: Optional group name for organizing runs
            tags: Optional list of tags
            config: Optional config dict to log
            notes: Optional notes for the run
            run_id: Optional run ID for resuming
            resume: Resume mode ("allow", "must", "never")
            reinit: Whether to allow reinit if a run exists
            job_type: Optional job type (e.g., "training", "attack")

        Returns:
            wandb.Run object if enabled, None otherwise
        """
        if not self.enabled:
            return None

        # Finish any existing run
        if self.current_run is not None:
            self.finish_run()

        try:
            self.current_run = wandb.init(
                project=self.project,
                name=name,
                group=group,
                tags=tags,
                config=config,
                notes=notes,
                id=run_id,
                resume=resume,
                reinit=reinit,
                job_type=job_type,
                dir="/mnt/data0/jayoung/INR-Robustness/wandb/"
            )
            return self.current_run
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to initialize wandb run: {e}")
            self.enabled = False
            return None

    def finish_run(self, quiet: bool = False):
        """Finish the current wandb run and cleanup.

        Args:
            quiet: Whether to suppress finish messages
        """
        if not self.enabled:
            return

        if self.current_run is not None:
            try:
                wandb.finish(quiet=quiet)
            except Exception as e:
                if not quiet:
                    print(f"[WandbManager] Warning: Error finishing run: {e}")
            finally:
                self.current_run = None

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """Log metrics to current wandb run.

        No-op if wandb is disabled or no active run.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
            commit: Whether to commit the log immediately
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to log metrics: {e}")

    def log_summary(self, summary: Dict[str, Any]):
        """Log summary metrics (shown in run table).

        Args:
            summary: Dictionary of summary metrics
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            for key, value in summary.items():
                wandb.run.summary[key] = value
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to log summary: {e}")

    def log_config(self, config: Dict[str, Any]):
        """Update the run config after initialization.

        Args:
            config: Dictionary of config values to update
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            wandb.config.update(config, allow_val_change=True)
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to update config: {e}")

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ):
        """Log an artifact (e.g., model checkpoint).

        Args:
            name: Artifact name
            artifact_type: Type (e.g., "model", "dataset")
            path: Path to artifact file/directory
            metadata: Optional metadata dict
            aliases: Optional list of aliases (e.g., ["best", "latest"])
        """
        if not self.enabled or self.current_run is None:
            return

        if not os.path.exists(path):
            print(f"[WandbManager] Warning: Artifact path does not exist: {path}")
            return

        try:
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
            if os.path.isfile(path):
                artifact.add_file(path)
            else:
                artifact.add_dir(path)
            wandb.log_artifact(artifact, aliases=aliases)
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to log artifact: {e}")

    def log_image(
        self,
        key: str,
        image: Any,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """Log an image to wandb.

        Args:
            key: Metric key for the image
            image: Image (PIL Image, numpy array, torch tensor, or path)
            caption: Optional caption
            step: Optional step number
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to log image: {e}")

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ):
        """Log a table to wandb.

        Args:
            key: Metric key for the table
            columns: List of column names
            data: List of rows (each row is a list of values)
            step: Optional step number
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({key: table}, step=step)
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to log table: {e}")

    @contextmanager
    def run(
        self,
        name: str,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Context manager for wandb run lifecycle.

        Automatically handles init and finish, even on exceptions.

        Usage:
            with manager.run(name="my_run", config=my_config):
                # training code
                manager.log({"loss": 0.5})

        Args:
            name: Run name
            group: Optional group
            tags: Optional tags
            config: Optional config
            **kwargs: Additional init_run arguments

        Yields:
            wandb.Run or None
        """
        run = self.init_run(
            name=name,
            group=group,
            tags=tags,
            config=config,
            **kwargs,
        )
        try:
            yield run
        finally:
            self.finish_run()

    @property
    def is_active(self) -> bool:
        """Check if there's an active run."""
        return self.enabled and self.current_run is not None

    def get_run_id(self) -> Optional[str]:
        """Get current run ID for checkpoint saving."""
        if not self.enabled or self.current_run is None:
            return None
        return wandb.run.id

    def get_run_url(self) -> Optional[str]:
        """Get URL to the current run in wandb dashboard."""
        if not self.enabled or self.current_run is None:
            return None
        try:
            return wandb.run.get_url()
        except Exception:
            return None

    def watch(
        self,
        model,
        log: str = "gradients",
        log_freq: int = 100,
        idx: Optional[int] = None,
    ):
        """Watch a model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all", or None)
            log_freq: How often to log
            idx: Optional model index for multi-model runs
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq, idx=idx)
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to watch model: {e}")

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ):
        """Send an alert (useful for long-running experiments).

        Args:
            title: Alert title
            text: Alert text
            level: Alert level ("INFO", "WARN", "ERROR")
        """
        if not self.enabled or self.current_run is None:
            return

        try:
            level_map = {
                "INFO": wandb.AlertLevel.INFO,
                "WARN": wandb.AlertLevel.WARN,
                "ERROR": wandb.AlertLevel.ERROR,
            }
            wandb.alert(
                title=title,
                text=text,
                level=level_map.get(level, wandb.AlertLevel.INFO),
            )
        except Exception as e:
            print(f"[WandbManager] Warning: Failed to send alert: {e}")

    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"WandbManager(enabled={self.enabled}, suite={self.suite_name}, status={status})"
