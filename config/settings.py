"""
Centralized configuration for the RSA Hardness Prediction Framework.

All constants, paths, and experimental parameters live here.
Two profiles are supported:
    - "full":         Original 100K samples (recommended 16GB RAM)
    - "memory_safe":  50K samples, chunked processing (for 8GB RAM)

Change PROFILE below to switch between them.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
# PROFILE SELECTOR — Change this based on your hardware
# ══════════════════════════════════════════════════════════════
PROFILE = "memory_safe"   # Options: "full" or "memory_safe"

# ══════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════
SEED = 42

# ══════════════════════════════════════════════════════════════
# Project paths (auto-resolved relative to project root)
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW    = PROJECT_ROOT / "data" / "raw"
DATA_PROC   = PROJECT_ROOT / "data" / "processed"
DATA_META   = PROJECT_ROOT / "data" / "meta"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_FIG = PROJECT_ROOT / "results" / "figures"
RESULTS_TAB = PROJECT_ROOT / "results" / "tables"
RESULTS_LOG = PROJECT_ROOT / "results" / "logs"

# Ensure output directories exist
for _d in [DATA_RAW, DATA_PROC, DATA_META, MODELS_DIR,
           RESULTS_FIG, RESULTS_TAB, RESULTS_LOG]:
    _d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Dataset generation parameters
# ══════════════════════════════════════════════════════════════
if PROFILE == "full":
    BIT_RANGES = {
        (32, 64):   40_000,
        (64, 96):   30_000,
        (96, 128):  20_000,
        (128, 160): 10_000,
    }
    BENCHMARK_BATCH_SIZE = 500
    FEATURE_CHUNK_SIZE   = 20_000
    TRAIN_BATCH_SIZE     = 1024
    TIMEOUT_SECONDS      = 300

elif PROFILE == "memory_safe":
    # 20K samples — lean but 200x larger than any reference paper
    BIT_RANGES = {
        (32, 64):   8_000,
        (64, 96):   6_000,
        (96, 128):  4_000,
        (128, 160): 2_000,
    }
    BENCHMARK_BATCH_SIZE = 250
    FEATURE_CHUNK_SIZE   = 10_000
    TRAIN_BATCH_SIZE     = 512
    TIMEOUT_SECONDS      = 5

else:
    raise ValueError(f"Unknown PROFILE: {PROFILE!r}. Use 'full' or 'memory_safe'.")

TOTAL_SAMPLES = sum(BIT_RANGES.values())

# Factor-structure distribution within each bit range
STRUCTURE_RATIOS = {
    "balanced":   0.40,   # p ≈ q (within a few bits)
    "moderate":   0.30,   # one factor 60–80% bit-length of the other
    "unbalanced": 0.20,   # one factor < 50% bit-length of the other
    "special":    0.10,   # Fermat-close, p-1 smooth, etc.
}

# ══════════════════════════════════════════════════════════════
# Benchmarking parameters
# ══════════════════════════════════════════════════════════════
BENCHMARK_ALGORITHMS = [
    "pollard_rho",
    "pollard_pm1",
    "ecm",
    "fermat",
]

# ══════════════════════════════════════════════════════════════
# Feature engineering parameters
# ══════════════════════════════════════════════════════════════
SMALL_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29]
POWER_MODS   = [256, 512, 1024]
JACOBI_BASES = [2, 3, 5, 7, 11]

# ══════════════════════════════════════════════════════════════
# Model training parameters
# ══════════════════════════════════════════════════════════════
TEST_SIZE  = 0.20
N_CV_FOLDS = 5
EARLY_STOP = 50

XGBOOST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.06,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "tree_method":      "hist",      # memory-efficient
    "random_state":     SEED,
}

CATBOOST_PARAMS = {
    "iterations":       400,
    "depth":            6,
    "learning_rate":    0.06,
    "l2_leaf_reg":      3.0,
    "verbose":          False,
    "random_state":     SEED,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64, 32),
    "activation":         "relu",
    "learning_rate_init": 1e-3,
    "max_iter":           80,
    "early_stopping":     True,
    "validation_fraction": 0.1,
    "n_iter_no_change":   10,
    "random_state":       SEED,
    "verbose":            False,
}

DUALBRANCH_SPLIT = {
    # Branch 1: size and entropy features
    "branch_1": [
        "log2_N", "bit_length",
        "hamming_weight", "hw_ratio",
        "binary_entropy", "byte_entropy",
        "longest_run_1s", "longest_run_0s", "bit_transitions",
    ],
    # Branch 2: modular residues + structural features (resolved at runtime)
    "branch_2": "ALL_MOD_AND_STRUCTURAL",
}

ELASTICNET_PARAMS = {
    "alpha":        0.01,
    "l1_ratio":     0.5,
    "max_iter":     10_000,
    "random_state": SEED,
}

# ══════════════════════════════════════════════════════════════
# PQC migration (NIST SP 800-57 Rev. 5)
# ══════════════════════════════════════════════════════════════
NIST_RSA_SECURITY = {
    # RSA modulus bits → classical security bits (NIST equivalence)
    1024:    80,
    2048:   112,
    3072:   128,
    4096:   140,     # interpolated
    7680:   192,
    15360:  256,
}

PQC_LEVELS = {
    1: {"security_bits": 128, "schemes": "Kyber-512 / Dilithium-2"},
    3: {"security_bits": 192, "schemes": "Kyber-768 / Dilithium-3"},
    5: {"security_bits": 256, "schemes": "Kyber-1024 / Dilithium-5"},
}

# Migration urgency weights (sum to 1.0)
URGENCY_WEIGHTS = {
    "vulnerability":   0.50,
    "security_margin": 0.30,
    "time_horizon":    0.20,
}

# ══════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════
LOG_LEVEL  = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s"


# ══════════════════════════════════════════════════════════════
# Convenience helper — run this file directly to see settings
# ══════════════════════════════════════════════════════════════
def print_config_summary() -> None:
    """Print the active configuration for verification."""
    print("═" * 60)
    print(f"  CONFIG PROFILE: {PROFILE.upper()}")
    print("═" * 60)
    print(f"  Total samples:        {TOTAL_SAMPLES:,}")
    print(f"  Bit ranges:           {dict(BIT_RANGES)}")
    print(f"  Benchmark timeout:    {TIMEOUT_SECONDS}s")
    print(f"  Benchmark batch:      {BENCHMARK_BATCH_SIZE}")
    print(f"  Feature chunk:        {FEATURE_CHUNK_SIZE:,}")
    print(f"  Train batch:          {TRAIN_BATCH_SIZE}")
    print(f"  CV folds:             {N_CV_FOLDS}")
    print(f"  Seed:                 {SEED}")
    print(f"  Project root:         {PROJECT_ROOT}")
    print("═" * 60)


if __name__ == "__main__":
    print_config_summary()