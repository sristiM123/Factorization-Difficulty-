"""
utils.py — Reproducibility, logging, I/O, and memory monitoring helpers.

Common utilities used throughout the framework. Keep this module
lightweight — no heavy ML imports.
"""

import datetime
import gc
import hashlib
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════
def set_global_seed(seed: int = 42) -> None:
    """Seed all random-number generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ══════════════════════════════════════════════════════════════
# Environment capture (for reproducibility metadata)
# ══════════════════════════════════════════════════════════════
def get_environment_info() -> Dict[str, str]:
    """Capture Python version, OS, and package versions for logging."""
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python":    platform.python_version(),
        "platform":  platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine":   platform.machine(),
    }
    for pkg in ("numpy", "pandas", "sklearn", "sympy",
                "xgboost", "catboost", "shap"):
        try:
            mod_name = "sklearn" if pkg == "sklearn" else pkg
            mod = __import__(mod_name)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "not installed"
    return info


def save_metadata(path: Path, meta: Dict[str, Any]) -> None:
    """Save a metadata dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def dataframe_checksum(df: pd.DataFrame) -> str:
    """Compute a deterministic MD5 hash of a DataFrame for integrity checks."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()


# ══════════════════════════════════════════════════════════════
# Memory monitoring — important for 8GB RAM systems
# ══════════════════════════════════════════════════════════════
def get_memory_usage_mb() -> float:
    """Return current process memory (RSS) in MB, or -1 if psutil unavailable."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return -1.0


def free_memory() -> None:
    """Force garbage collection and release memory back to the OS."""
    gc.collect()
    try:
        import ctypes
        if hasattr(ctypes, "CDLL"):
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except (OSError, AttributeError):
                pass
    except ImportError:
        pass


def log_memory(label: str = "") -> None:
    """Print current memory usage with an optional label."""
    mb = get_memory_usage_mb()
    if mb > 0:
        print(f"  [MEM] {label}: {mb:,.1f} MB")


# ══════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════
def get_logger(
        name: str,
        log_file: Optional[Path] = None,
        level: str = "INFO",
        fmt: str = "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
) -> logging.Logger:
    """Create a configured logger with optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_file, encoding="utf-8")
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)

    return logger


# ══════════════════════════════════════════════════════════════
# Presentation helpers
# ══════════════════════════════════════════════════════════════
def print_section(title: str, char: str = "═", width: int = 70) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def load_processed_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience loader for the final processed dataset."""
    if path is None:
        from config import DATA_PROC
        path = DATA_PROC / "dataset_with_features_and_runtimes.csv"
    return pd.read_csv(path)