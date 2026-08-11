"""Per-epoch logging that mirrors nnU-Net's, including the EMA pseudo-Dice.

nnU-Net does **not** checkpoint on a full-image validation Dice. It tracks a *pseudo*
Dice — global TP/FP/FN accumulated over 50 random validation patches per epoch, per
class — and smooths it with a 0.9 EMA; ``checkpoint_best`` is whatever epoch maximised
that EMA. Reproducing the checkpoint-selection rule matters as much as reproducing the
optimiser, because it decides which weights the final numbers come from.

Also writes nnU-Net's ``progress.png`` (loss curves + pseudo-Dice + LR + epoch time) and
mirrors every scalar into the repo's TensorBoard tracker so replica runs sit next to
exp01–19 in the same board.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class ReplicaLogger:
    """One value per key per epoch, plus the derived EMA pseudo-Dice."""

    KEYS = (
        "mean_fg_dice", "ema_fg_dice", "dice_per_class", "train_losses",
        "val_losses", "lrs", "epoch_start_timestamps", "epoch_end_timestamps",
    )

    def __init__(self, output_folder: str | Path, tracker=None):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.logging: Dict[str, List] = {k: [] for k in self.KEYS}
        self.tracker = tracker
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.log_file = self.output_folder / f"training_log_{stamp}.txt"

    # ---------------------------------------------------------------- logging
    def log(self, key: str, value, epoch: int) -> None:
        assert key in self.logging, f"unknown log key '{key}'"
        series = self.logging[key]
        if len(series) < epoch + 1:
            series.extend([None] * (epoch + 1 - len(series)))
        series[epoch] = value

        if key == "mean_fg_dice":
            prev = self.logging["ema_fg_dice"]
            ema = prev[epoch - 1] * 0.9 + 0.1 * value if epoch > 0 and prev and prev[epoch - 1] is not None else value
            self.log("ema_fg_dice", ema, epoch)

        if self.tracker is not None and isinstance(value, (int, float, np.floating)):
            self.tracker.log_scalar(f"replica/{key}", float(value), epoch)

    def print_to_log_file(self, *args, add_timestamp: bool = True, also_print: bool = True) -> None:
        parts = [str(a) for a in args]
        line = " ".join(parts)
        if add_timestamp:
            line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {line}"
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
        if also_print:
            print(line, flush=True)

    # ------------------------------------------------------------ persistence
    def save_json(self, filename: str = "training_history.json") -> None:
        payload = {k: [None if v is None else (v.tolist() if isinstance(v, np.ndarray) else v)
                       for v in series]
                   for k, series in self.logging.items()}
        with open(self.output_folder / filename, "w") as f:
            json.dump(payload, f, indent=2, default=float)

    def load_json(self, filename: str = "training_history.json") -> None:
        path = self.output_folder / filename
        if path.is_file():
            with open(path) as f:
                loaded = json.load(f)
            for k in self.KEYS:
                if k in loaded:
                    self.logging[k] = loaded[k]

    # ------------------------------------------------------------------ plots
    def plot_progress_png(self, filename: str = "progress.png") -> None:
        """nnU-Net-style three-panel progress plot (losses / pseudo-Dice / LR + epoch time)."""
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        n = min(len(self.logging[k]) for k in ("train_losses", "val_losses", "lrs"))
        if n < 1:
            return
        x = list(range(n))

        fig, axes = plt.subplots(3, 1, figsize=(18, 24))

        ax = axes[0]
        ax.plot(x, self.logging["train_losses"][:n], color="b", ls="-", label="loss_tr")
        ax.plot(x, self.logging["val_losses"][:n], color="r", ls="-", label="loss_val")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(loc="upper right")

        ax = axes[1]
        dice = self.logging["mean_fg_dice"][:n]
        ema = self.logging["ema_fg_dice"][:n]
        if any(v is not None for v in dice):
            ax.plot(x, dice, color="g", ls="dotted", label="pseudo dice")
            ax.plot(x, ema, color="g", ls="-", label="pseudo dice (EMA)")
        ax.set_xlabel("epoch"); ax.set_ylabel("pseudo Dice"); ax.legend(loc="lower right")

        ax = axes[2]
        ax.plot(x, self.logging["lrs"][:n], color="b", ls="-", label="learning rate")
        ax.set_xlabel("epoch"); ax.set_ylabel("lr"); ax.legend(loc="upper right")
        starts, ends = self.logging["epoch_start_timestamps"], self.logging["epoch_end_timestamps"]
        if len(starts) >= n and len(ends) >= n and all(
            starts[i] is not None and ends[i] is not None for i in range(n)
        ):
            ax2 = ax.twinx()
            ax2.plot(x, [ends[i] - starts[i] for i in range(n)], color="0.5", ls="-",
                     label="epoch time (s)")
            ax2.set_ylabel("epoch time (s)"); ax2.legend(loc="lower right")

        plt.tight_layout()
        fig.savefig(self.output_folder / filename)
        plt.close(fig)

    # ---------------------------------------------------------------- getters
    def latest(self, key: str) -> Optional[float]:
        series = [v for v in self.logging[key] if v is not None]
        return series[-1] if series else None
