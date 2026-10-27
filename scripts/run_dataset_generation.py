"""
run_dataset_generation.py — Standalone Phase 1 pipeline.

Usage:
    python scripts/run_dataset_generation.py

Generates semiprimes, extracts features, benchmarks runtimes, and saves
the processed dataset. Use this for unattended batch runs.

For interactive exploration with plots, use:
    notebooks/01_dataset_generation.ipynb
"""

import sys
from pathlib import Path

# Make project root importable regardless of where this script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import (
    SEED, BIT_RANGES, STRUCTURE_RATIOS, TOTAL_SAMPLES,
    TIMEOUT_SECONDS, BENCHMARK_ALGORITHMS,
    BENCHMARK_BATCH_SIZE, FEATURE_CHUNK_SIZE,
    SMALL_PRIMES, POWER_MODS, JACOBI_BASES,
    DATA_RAW, DATA_PROC, DATA_META, RESULTS_LOG,
    print_config_summary,
)
from src.generators import generate_prime_pool, generate_semiprimes
from src.features import extract_features, get_feature_names
from src.benchmarks import benchmark_batch
from src.utils import (
    set_global_seed, get_environment_info, save_metadata,
    dataframe_checksum, print_section, log_memory, free_memory,
    get_logger,
)


def main():
    # ─── Setup ────────────────────────────────────────────
    set_global_seed(SEED)
    log = get_logger("phase1", RESULTS_LOG / "phase1_dataset.log")

    print_config_summary()
    log_memory("start")

    # ─── Phase 1.1: Prime pool ───────────────────────────
    print_section("PHASE 1.1 — Prime Pool Generation")
    max_bits = max(hi for (lo, hi) in BIT_RANGES.keys())
    prime_pool = generate_prime_pool(max_bits=max_bits)
    total_primes = sum(len(v) for v in prime_pool.values())
    log.info(f"Prime pool: {total_primes:,} primes, "
             f"{len(prime_pool)} bit-lengths")
    log_memory("after prime pool")

    # ─── Phase 1.2: Semiprime generation ─────────────────
    print_section("PHASE 1.2 — Semiprime Generation")
    records = generate_semiprimes(
        bit_ranges=BIT_RANGES,
        structure_ratios=STRUCTURE_RATIOS,
        prime_pool=prime_pool,
        seed=SEED,
    )
    df_raw = pd.DataFrame([{
        "N": r.N, "bit_length": r.bit_length,
        "p_hidden": r.p, "q_hidden": r.q,
        "structure": r.structure,
        "p_bits": r.p_bits, "q_bits": r.q_bits,
        "balance_ratio": r.balance_ratio,
    } for r in records])

    df_raw.to_csv(DATA_RAW / "semiprimes_raw.csv", index=False)
    log.info(f"Raw dataset saved: {len(df_raw):,} semiprimes")

    # Free memory: prime pool no longer needed
    del prime_pool, records
    free_memory()
    log_memory("after raw save")

    # ─── Phase 1.3: Feature engineering ──────────────────
    print_section("PHASE 1.3 — Feature Engineering")
    df_features = extract_features(
        df_raw["N"].tolist(),
        small_primes=SMALL_PRIMES,
        power_mods=POWER_MODS,
        jacobi_bases=JACOBI_BASES,
        chunk_size=FEATURE_CHUNK_SIZE,
    )
    log.info(f"Features extracted: {df_features.shape}")
    log_memory("after features")

    # ─── Phase 1.4: Runtime benchmarking ─────────────────
    print_section("PHASE 1.4 — Runtime Benchmarking")
    bench_results = benchmark_batch(
        df_raw["N"].tolist(),
        algorithms=BENCHMARK_ALGORITHMS,
        timeout=TIMEOUT_SECONDS,
        batch_size=BENCHMARK_BATCH_SIZE,
    )
    df_bench = pd.DataFrame(bench_results)
    log.info(f"Benchmarks complete: {df_bench.shape}")
    log_memory("after benchmarks")

    # ─── Phase 1.5: Merge and save ───────────────────────
    print_section("PHASE 1.5 — Merge & Save")
    df_final = pd.concat([
        df_raw.reset_index(drop=True),
        df_features.reset_index(drop=True),
        df_bench.drop(columns=["N"]).reset_index(drop=True),
    ], axis=1)

    feature_names = get_feature_names(SMALL_PRIMES, POWER_MODS)
    public_cols = (
            ["N", "bit_length", "structure"]
            + feature_names
            + [f"runtime_{a}" for a in BENCHMARK_ALGORITHMS]
            + [f"success_{a}" for a in BENCHMARK_ALGORITHMS]
            + [f"timeout_{a}" for a in BENCHMARK_ALGORITHMS]
            + ["best_algorithm", "min_runtime", "log_min_runtime"]
    )

    df_final[public_cols].to_csv(
        DATA_PROC / "dataset_with_features_and_runtimes.csv", index=False,
        )
    df_final[["N", "p_hidden", "q_hidden", "p_bits",
              "q_bits", "balance_ratio"]].to_csv(
        DATA_RAW / "hidden_factors.csv", index=False,
        )

    save_metadata(DATA_META / "feature_config.json", {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "target_regression": "log_min_runtime",
        "target_classification": "best_algorithm",
        "algorithms": BENCHMARK_ALGORITHMS,
    })

    save_metadata(DATA_META / "dataset_metadata.json", {
        "total_samples": len(df_final),
        "bit_ranges": {f"{lo}-{hi}": c for (lo, hi), c in BIT_RANGES.items()},
        "structure_ratios": STRUCTURE_RATIOS,
        "seed": SEED,
        "timeout_seconds": TIMEOUT_SECONDS,
        "checksum": dataframe_checksum(df_final[public_cols]),
        "environment": get_environment_info(),
    })

    log.info(f"Phase 1 complete. Final shape: {df_final[public_cols].shape}")
    log_memory("end")


if __name__ == "__main__":
    main()