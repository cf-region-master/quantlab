from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .pool import AlphaPool, Calculator


class ResourceMonitor:
    def __init__(self, path: Path, interval: float = 2.0, append: bool = False):
        self.path, self.interval, self.append = path, interval, append
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval * 2))

    def _gpu(self) -> tuple[float, float, float]:
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits", "--id=0"
            ], capture_output=True, text=True, timeout=5, check=True)
            return tuple(map(float, result.stdout.strip().split(", ")))  # type: ignore[return-value]
        except Exception: return float("nan"), float("nan"), float("nan")

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if self.append and exists else "w"
        with self.path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(["elapsed_sec", "gpu_util_pct", "gpu_memory_mb", "gpu_memory_total_mb",
                                 "torch_allocated_mb", "cpu_pct", "rss_mb"])
            start = time.monotonic()
            while not self.stop_event.is_set():
                util, used, total = self._gpu()
                allocated = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
                writer.writerow([time.monotonic() - start, util, used, total, allocated,
                                 psutil.cpu_percent(), psutil.Process().memory_info().rss / 2**20])
                f.flush()
                self.stop_event.wait(self.interval)


class ExperimentCallback(BaseCallback):
    def __init__(self, run_dir: Path, pool: AlphaPool, validation: Calculator):
        super().__init__(verbose=0)
        self.run_dir, self.pool, self.validation = run_dir, pool, validation
        self.round = 0
        self.metrics_path = run_dir / "metrics.csv"
        self.candidate_dir = run_dir / "candidates"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "round", "steps", "train_ic", "train_icir", "train_coverage",
                "validation_ic", "validation_icir", "validation_coverage",
                "round_repeat_rate", "history_repeat_rate", "avg_depth", "avg_length", "entropy",
                "candidates", "valid_factor_fps", "valid_factor_bar_fps",
                "factor_eval_seconds", "pool_size"
            ])

    def _on_step(self) -> bool: return True

    def _on_rollout_end(self) -> None:
        self.round += 1
        events = self.pool.drain_events()
        path = self.candidate_dir / f"round_{self.round:05d}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["token", "ic"])
            writer.writerows((e.token, e.ic) for e in events)
        tokens = [e.token for e in events]
        n = len(events)
        train = self.pool.metrics(exact=True)
        valid = self.pool.metrics(self.validation, exact=True)
        logger_values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = logger_values.get("train/entropy_loss", float("nan"))
        evaluated, eval_seconds = self.pool.drain_eval_stats()
        valid_factor_fps = evaluated / max(1e-9, eval_seconds)
        bars_per_factor = sum(len(part.target) for part in self.pool.train.market.splits.values())
        valid_factor_bar_fps = evaluated * bars_per_factor / max(1e-9, eval_seconds)
        row = [
            self.round, self.num_timesteps, train["ic"], train["icir"], train["coverage"],
            valid["ic"], valid["icir"], valid["coverage"],
            1 - len(set(tokens)) / n if n else 0.0,
            sum(e.seen_before for e in events) / n if n else 0.0,
            np.mean([e.depth for e in events]) if n else float("nan"),
            np.mean([e.length for e in events]) if n else float("nan"),
            -float(entropy_loss), n, valid_factor_fps, valid_factor_bar_fps,
            eval_seconds, len(self.pool.entries),
        ]
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f: csv.writer(f).writerow(row)
        print(f"[round {self.round}] train_ic={train['ic']:.6f} val_ic={valid['ic']:.6f} "
              f"coverage={train['coverage']:.3f}/{valid['coverage']:.3f} "
              f"candidates={n} valid_factor_fps={valid_factor_fps:.3f} "
              f"factor_bar_fps={valid_factor_bar_fps:.0f}")


def save_pool(path: Path, pool: AlphaPool) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["token", "ic"])
        writer.writerows((e.token, e.single_ic) for e in pool.entries)


def _plot(lines, xlabel: str, ylabel: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for x, y, label in lines: ax.plot(x, y, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.25)
    if any(label for _, _, label in lines): ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_plots(run_dir: Path) -> None:
    import pandas as pd
    metrics = pd.read_csv(run_dir / "metrics.csv")
    if not metrics.empty:
        x = metrics["round"]
        ic_lines = [(x, metrics["train_ic"], "train"), (x, metrics["validation_ic"], "validation")]
        if "test_ic" in metrics: ic_lines.append((x, metrics["test_ic"], "test"))
        _plot(ic_lines,
              "round", "ensemble IC", run_dir / "figures/ic_curve.png")
        _plot([(x, metrics["train_coverage"], "train"),
               (x, metrics["validation_coverage"], "validation")],
              "round", "IC coverage", run_dir / "figures/coverage.png")
        _plot([(x, metrics["round_repeat_rate"], "within-round duplicate rate"),
               (x, metrics["history_repeat_rate"], "historical duplicate rate")],
              "round", "repeat rate", run_dir / "figures/repeat_rate.png")
        _plot([(x, metrics["avg_depth"], "average depth"), (x, metrics["avg_length"], "average length")],
              "round", "complexity", run_dir / "figures/depth_length.png")
        _plot([(x, metrics["entropy"], "entropy")], "round", "entropy",
              run_dir / "figures/entropy.png")
    resource_path = run_dir / "resource_usage.csv"
    if resource_path.exists():
        resource = pd.read_csv(resource_path)
        if not resource.empty:
            total = resource["gpu_memory_total_mb"].replace(0, np.nan)
            mem_pct = resource["gpu_memory_mb"] / total * 100
            _plot([(resource["elapsed_sec"], resource["gpu_util_pct"], "GPU utilization"),
                   (resource["elapsed_sec"], mem_pct, "GPU memory share")],
                  "elapsed seconds", "%", run_dir / "figures/gpu_utilization.png")
