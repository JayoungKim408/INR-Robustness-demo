"""
Results Collector for centralized CSV-based experiment tracking.

Provides unified schema for storing and querying experiment results across
all attack types, models, and datasets.
"""

import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading


class ResultsCollector:
    """Centralized CSV-based results aggregation system.

    Features:
    - Standardized schema across all experiment types
    - Thread-safe append operations
    - Query and filtering capabilities
    - Export to LaTeX tables
    """

    # Standard CSV schema columns
    BASE_COLUMNS = [
        # Metadata
        "experiment_id",
        "suite_name",
        "timestamp",

        # Dataset and model
        "dataset",
        "victim_model",

        # Attack configuration
        "attack_space",
        "attack_type",

        # INR configuration
        "inr_depth",
        "inr_width",
        "inr_scale",
        "inr_extra_aug",
        "inr_training_steps",

        # Core results
        "cls_acc",
        "surrogate_clean_acc",
        "surrogate_adv_acc",
        "clean_acc",
        "adv_acc",
        "total_samples",

        # Attack details
        "eps",
        "steps",
        "gamma",

        # Performance metrics
        "training_time_sec",
        "attack_time_sec",

        # Status
        "status",  # success, failed, running
        "error_msg",
        "checkpoint_path",
    ]

    def __init__(self, results_db_path: str):
        """Initialize results collector.

        Args:
            results_db_path: Path to CSV database file
        """
        self.results_db_path = results_db_path
        self._lock = threading.Lock()

        if os.path.exists(results_db_path):
            with open(results_db_path, 'r') as f:
                reader = csv.reader(f)
                self._columns = next(reader)
        else:
            self._columns = list(self.BASE_COLUMNS)
            self._create_database()

    def _create_database(self):
        """Create new CSV database with headers."""
        os.makedirs(os.path.dirname(self.results_db_path), exist_ok=True)

        with open(self.results_db_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()

    def _ensure_columns(self, new_keys: List[str]):
        added = [k for k in new_keys if k not in self._columns]
        if not added:
            return
        self._columns.extend(added)
        if os.path.exists(self.results_db_path):
            df = pd.read_csv(self.results_db_path)
            for col in added:
                df[col] = ""
            df.to_csv(self.results_db_path, index=False)  # columns= 인자 제거

    def append_result(self, exp_config: Dict[str, Any],
                     attack_result: Dict[str, Any]):
        """Append single experiment result to CSV database.

        Args:
            exp_config: Experiment configuration dict with keys:
                - experiment_id
                - suite_name
                - dataset
                - victim_model
                - attack_space
                - attack_type
                - inr_config (dict with depth, width, scale, etc.)
                - attack_params (dict with attack-specific params)
            attack_result: Attack result dict with keys:
                - clean_acc
                - adv_acc
                - total_samples
                - eps, steps, gamma (optional)
                - training_time_sec (optional)
                - attack_time_sec (optional)
                - status
                - error_msg (optional)
                - checkpoint_path (optional)
        """
        # Build row from configs and results
        row = {col: "" for col in self._columns}

        # Metadata
        row["experiment_id"] = exp_config.get("experiment_id", "")
        row["suite_name"] = exp_config.get("suite_name", "")
        row["timestamp"] = datetime.now().isoformat()

        # Dataset and model
        row["dataset"] = exp_config.get("dataset", "")
        row["victim_model"] = exp_config.get("victim_model", "")

        # Attack config
        row["attack_space"] = exp_config.get("attack_space", "")
        row["attack_type"] = exp_config.get("attack_type", "")

        # INR config
        inr_config = exp_config.get("inr_config", {})
        row["inr_depth"] = inr_config.get("depth", "")
        row["inr_width"] = inr_config.get("width", "")
        row["inr_scale"] = inr_config.get("scale", "")
        row["inr_extra_aug"] = inr_config.get("extra_aug", "")
        row["inr_training_steps"] = inr_config.get("training_steps", "")

        attack_params = exp_config.get("attack_params", {})
        for key, value in attack_params.items():
            row[key] = value

        # Core results
        row["clean_acc"] = attack_result.get("clean_acc", "")
        row["cls_acc"] = attack_result.get("cls_acc", "")
        row["surrogate_clean_acc"] = attack_result.get("surrogate_clean_acc", "")
        row["surrogate_adv_acc"] = attack_result.get("surrogate_adv_acc", "")
        row["adv_acc"] = attack_result.get("adv_acc", "")
        row["total_samples"] = attack_result.get("total", attack_result.get("total_samples", ""))

        # Attack details
        row["eps"] = attack_result.get("eps", "")
        row["steps"] = attack_result.get("steps", "")
        row["gamma"] = attack_result.get("gamma", "")

        # Performance
        row["training_time_sec"] = attack_result.get("training_time_sec", "")
        row["attack_time_sec"] = attack_result.get("attack_time_sec", "")

        # Status
        row["status"] = attack_result.get("status", "success")
        row["error_msg"] = attack_result.get("error_msg", "")
        row["checkpoint_path"] = attack_result.get("checkpoint_path", "")

        with self._lock:
            self._ensure_columns(list(row.keys()))  # 이 줄 추가
            with open(self.results_db_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._columns)
                writer.writerow(row)

    def get_results(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Query results with optional filters.

        Args:
            filters: Dictionary of column -> value filters.
                Example: {"victim_model": "nft", "dataset": "mnist"}

        Returns:
            Filtered DataFrame
        """
        if not os.path.exists(self.results_db_path):
            return pd.DataFrame(columns=self._columns)

        df = pd.read_csv(self.results_db_path)

        if filters:
            for column, value in filters.items():
                if column in df.columns:
                    df = df[df[column] == value]

        return df

    def get_summary(self, group_by: List[str],
                   metrics: List[str] = None) -> pd.DataFrame:
        """Generate summary table grouped by specified columns.

        Args:
            group_by: Columns to group by (e.g., ["victim_model", "attack_type"])
            metrics: Metrics to aggregate (default: ["clean_acc", "adv_acc"])

        Returns:
            Grouped summary DataFrame
        """
        if metrics is None:
            metrics = ["clean_acc", "adv_acc"]

        df = self.get_results()

        # Convert metric columns to numeric
        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')

        # Group and aggregate
        summary = df.groupby(group_by)[metrics].agg(['mean', 'std', 'count'])

        return summary

    def export_summary(self, output_path: str, group_by: List[str],
                      metrics: List[str] = None, format: str = 'csv'):
        """Export summary table for paper.

        Args:
            output_path: Output file path
            group_by: Columns to group by
            metrics: Metrics to include
            format: Output format ('csv', 'latex', 'markdown')
        """
        summary = self.get_summary(group_by, metrics)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == 'csv':
            summary.to_csv(output_path)
        elif format == 'latex':
            latex_str = summary.to_latex(float_format="%.4f")
            with open(output_path, 'w') as f:
                f.write(latex_str)
        elif format == 'markdown':
            md_str = summary.to_markdown()
            with open(output_path, 'w') as f:
                f.write(md_str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Exported summary to {output_path}")

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific experiment ID.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Dictionary with experiment result or None if not found
        """
        df = self.get_results(filters={"experiment_id": experiment_id})

        if len(df) == 0:
            return None

        return df.iloc[0].to_dict()

    def count_experiments(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count experiments matching filters.

        Args:
            filters: Optional filters

        Returns:
            Number of matching experiments
        """
        df = self.get_results(filters)
        return len(df)

    def get_status_counts(self, suite_name: Optional[str] = None) -> Dict[str, int]:
        """Get counts of experiments by status.

        Args:
            suite_name: Optional suite name filter

        Returns:
            Dictionary mapping status -> count
        """
        filters = {"suite_name": suite_name} if suite_name else None
        df = self.get_results(filters)

        if "status" in df.columns:
            return df["status"].value_counts().to_dict()
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return f"ResultsCollector(db={self.results_db_path})"
