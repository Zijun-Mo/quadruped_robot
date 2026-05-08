"""Utility helpers for wandb utils support in rl_base."""

from __future__ import annotations

import os


class WandbSummaryWriter:
    """Experiment logging adapter for wandb summary writer."""
    def __init__(self, log_dir: str, flush_secs: int = 10, cfg=None):
        """Initialize WandbSummaryWriter with configuration, tensor shapes, and runtime state."""
        self.log_dir = log_dir
        self.writer = None
        self.wandb = None
        self.run = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        except Exception:
            self.writer = None
        try:
            import wandb

            self.wandb = wandb
            cfg = cfg or {}
            project = cfg.get("wandb_project") or cfg.get("experiment_name") or "unitree_rl_lab"
            name = cfg.get("run_name") or os.path.basename(log_dir)
            entity = os.environ.get("WANDB_USERNAME") or os.environ.get("WANDB_ENTITY")
            self.run = wandb.init(project=project, entity=entity, name=name, dir=log_dir, config=cfg, reinit=True)
        except Exception as exc:
            print(f"[WARN] wandb disabled: {exc}")
            self.wandb = None
            self.run = None

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Store run configuration metadata for experiment tracking."""
        return self.log_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        """Log a scalar metric value to the backing writer."""
        value = float(scalar_value)
        if self.writer is not None:
            self.writer.add_scalar(tag, value, global_step, walltime, new_style)
        if self.wandb is not None:
            try:
                self.wandb.log({tag: value}, step=global_step)
            except Exception:
                pass

    def stop(self):
        """Flush and close the experiment writer."""
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Log configuration files or dictionaries for experiment tracking."""
        if self.run is not None:
            try:
                self.run.config.update(
                    {"env_cfg": str(env_cfg), "runner_cfg": runner_cfg, "alg_cfg": alg_cfg, "policy_cfg": policy_cfg},
                    allow_val_change=True,
                )
            except Exception:
                pass

    def save_model(self, model_path, iter):
        """Persist a model checkpoint through the experiment writer."""
        self.save_file(model_path, iter)

    def save_file(self, path, iter=None):
        """Persist an auxiliary file through the experiment writer."""
        if self.wandb is not None:
            try:
                self.wandb.save(path)
            except Exception:
                pass

    @staticmethod
    def _map_path(path):
        """Map a filesystem path into the logger-specific namespace."""
        return str(path).replace("/", "_")
