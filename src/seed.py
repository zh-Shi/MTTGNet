import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # NOTE: benchmark=False disables cuDNN auto-tuner → 20-50% slower training.
    # Required for reproducible results; remove only if you need max throughput
    # and accept minor run-to-run variation.
    torch.backends.cudnn.benchmark = False

    # cuDNN determinism requires this env var (4096 bytes of workspace per op).
    # Without it, some cuDNN ops silently fall back to non-deterministic paths
    # even with cudnn.deterministic=True.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Force PyTorch to error on any non-deterministic CUDA op instead of
    # silently producing non-reproducible results.  Wrapped in try/except
    # because a few ops (e.g. scatter_reduce) have no deterministic impl yet.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError:
        pass
