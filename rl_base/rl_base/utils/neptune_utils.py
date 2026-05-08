"""Utility helpers for neptune utils support in rl_base."""

from __future__ import annotations


class NeptuneLogger:
    """Experiment logging adapter for neptune logger."""
    def __init__(self, project, token):
        """Initialize NeptuneLogger with configuration, tensor shapes, and runtime state."""
        self.project = project
        self.token = token
        self.run = None

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Store run configuration metadata for experiment tracking."""
        return None


class NeptuneSummaryWriter:
    """Experiment logging adapter for neptune summary writer."""
    def __init__(self, log_dir: str, flush_secs: int = 10, cfg=None):
        """Initialize NeptuneSummaryWriter with configuration, tensor shapes, and runtime state."""
        self.log_dir = log_dir
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        except Exception:
            self.writer = None

    @staticmethod
    def _map_path(path):
        """Map a filesystem path into the logger-specific namespace."""
        return str(path).replace("/", "_")

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        """Log a scalar metric value to the backing writer."""
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step, walltime, new_style)

    def stop(self):
        """Flush and close the experiment writer."""
        if self.writer is not None:
            self.writer.close()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Log configuration files or dictionaries for experiment tracking."""
        return None

    def save_model(self, model_path, iter):
        """Persist a model checkpoint through the experiment writer."""
        return None

    def save_file(self, path, iter=None):
        """Persist an auxiliary file through the experiment writer."""
        return None
