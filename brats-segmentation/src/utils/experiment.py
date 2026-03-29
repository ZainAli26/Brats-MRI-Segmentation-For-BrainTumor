"""Experiment tracking: config management, run naming, logging, and result persistence."""

import json
import shutil
import yaml
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from rich.console import Console

console = Console()


class ExperimentTracker:
    """Manages experiment runs: directories, configs, TensorBoard, and result logs."""

    def __init__(self, config: dict, config_path: str = None):
        model_name = config["model"]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"

        output_dir = Path(config["experiment"]["output_dir"]).expanduser()
        self.run_dir = output_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.viz_dir = self.run_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Save config
        config_save_path = self.run_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Copy original config file if provided
        if config_path:
            shutil.copy2(config_path, self.run_dir / "config_original.yaml")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

        # Text log
        self.log_file = self.run_dir / "training.log"

        self.run_name = run_name
        self.config = config

        console.print(f"[bold green]Experiment: {run_name}[/bold green]")
        console.print(f"[dim]Run directory: {self.run_dir}[/dim]")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar to TensorBoard and text log."""
        self.writer.add_scalar(tag, value, step)
        with open(self.log_file, "a") as f:
            f.write(f"{tag},{step},{value}\n")

    def log_text(self, message: str):
        """Log a text message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def save_metrics(self, metrics_df, filename: str = "metrics.csv"):
        """Save evaluation metrics DataFrame."""
        path = self.run_dir / filename
        metrics_df.to_csv(path, index=False)
        console.print(f"[green]Saved metrics to {path}[/green]")

    def save_summary(self, summary: dict, filename: str = "summary.json"):
        """Save a summary dict as JSON."""
        path = self.run_dir / filename
        # Convert numpy types for JSON serialization
        clean = {}
        for k, v in summary.items():
            if hasattr(v, 'item'):
                clean[k] = v.item()
            elif isinstance(v, dict):
                clean[k] = {kk: vv.item() if hasattr(vv, 'item') else vv for kk, vv in v.items()}
            else:
                clean[k] = v
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        console.print(f"[green]Saved summary to {path}[/green]")

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
