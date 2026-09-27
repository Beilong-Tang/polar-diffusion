"""
Logging, distributed setup and small helpers shared by the training and
sampling scripts.
"""
import argparse
import datetime
import glob
import logging
import os
import random
import time
from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v!r}")


def get_value(name, default, config: dict, args: argparse.Namespace):
    """Return args.<name> if the parser defines it, else config[name], else default."""
    if name in vars(args):
        return vars(args)[name]
    return config.get(name, default)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Thin wrapper around `logging` that only prints from rank 0 by default."""

    def __init__(self, log: logging.Logger, rank: int):
        self.log = log
        self.rank = rank

    def __call__(self, msg):
        self.info(msg)

    def info(self, msg: str, all=False):
        if all or self.rank == 0:
            self.log.info(msg)

    def debug(self, msg: str):
        if self.rank == 0:
            self.log.debug(msg)

    def warning(self, msg: str):
        self.log.warning(f"rank {self.rank} - {msg}")

    def error(self, msg: str):
        self.log.error(f"rank {self.rank} - {msg}")


def setup_logger(log_dir, rank=0, out=True):
    """Log to stdout and, if `out`, to <log_dir>/<timestamp>.log."""
    handlers = [logging.StreamHandler()]
    if out and log_dir is not None:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, f"{now}.log")))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger()
    logger.info("logger initialized")
    return Logger(logger, rank)


# ---------------------------------------------------------------------------
# Distributed (torchrun) setup
# ---------------------------------------------------------------------------

def setup_seed(seed, rank=0):
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def setup_torchrun(seed, log_dir, init_process=True, log_to_file=True):
    """
    Set up a process launched by torchrun (or a plain single process).

    init_process: create the NCCL process group. Sampling only needs the
        rank/world size to split work, so it can skip this.
    returns (rank, world_size, device, logger)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if init_process:
            dist.init_process_group(backend="nccl")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed, rank)
    logger = setup_logger(log_dir, rank, out=log_to_file)
    return rank, world_size, device, logger


def ddp_cleanup(func):
    """Destroy the process group when `func` exits, even on error."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    return wrapper


def pending_sample_indices(num_samples, rank, world_size, img_dir, suffix="png"):
    """
    Indices of the images this rank still has to generate.

    Indices are split across ranks as rank::world_size; images already saved in
    img_dir as <index>.png or <index>_*.png are skipped, so an interrupted run
    can be resumed by re-running the same command.
    """
    ids = list(range(num_samples))[rank::world_size]
    done = {int(Path(p).stem.split("_")[0]) for p in glob.glob(os.path.join(img_dir, f"*.{suffix}"))}
    return [i for i in ids if i not in done]


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def gpu_stats(device=None):
    """Short GPU utilization/memory string; empty if pynvml is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return ""
    if device is None:
        device = torch.cuda.current_device()
    if isinstance(device, torch.device):
        device = device.index or 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return f"[GPU Util: {util.gpu}% | Mem: {100 * mem.used / mem.total:.1f}%]"


def _time_to_str(t):
    return str(datetime.timedelta(seconds=int(t)))


class ContextTimer:
    """Step timer with per-section averages and an ETA, e.g. `with timer.track("data"):`."""

    def __init__(self, total_steps, avg_window=10):
        self.total_steps = total_steps
        self.step_count = 0
        self.timers = {}
        self.avg_window = avg_window
        self.step_history = deque(maxlen=avg_window)
        self.total_step_time = 0.0
        self.step_start_time = time.time()

    @contextmanager
    def track(self, key):
        t0 = time.time()
        yield
        self.timers.setdefault(key, deque(maxlen=self.avg_window)).append(time.time() - t0)

    def update(self):
        elapsed = time.time() - self.step_start_time
        self.total_step_time += elapsed
        self.step_count += 1
        self.step_history.append(elapsed)
        self.step_start_time = time.time()

    def stats(self):
        per_step = sum(self.step_history) / len(self.step_history) if self.step_history else 0.0
        eta = _time_to_str((self.total_steps - self.step_count) * per_step)
        res = f"[{_time_to_str(self.total_step_time)}<{eta}, {per_step:.3f}s/it"
        for k, v in self.timers.items():
            if v:
                res += f", {k}: {sum(v) / len(v):.3f}s"
        return res + "]"
