"""Tests for src/benchmarks.py — verifies factorization timing and result structure."""

import pytest
from src.benchmarks import (
    pollard_rho,
    benchmark_pollard_rho,
    benchmark_ecm,
    benchmark_pollard_pm1,
    benchmark_fermat,
    benchmark_semiprime,
    benchmark_batch,
    BenchmarkResult,
    ALGORITHM_MAP,
)


# ══════════════════════════════════════════════════════════════
# Pollard's rho algorithm (core)
# ══════════════════════════════════════════════════════════════
class TestPollardRho:

    def test_small_semiprime(self):
        # 15 = 3 × 5
        factor = pollard_rho(15, max_iterations=1000)
        assert factor in [3, 5]

    def test_correct_factorization(self):
        # 1001 = 7 × 11 × 13
        factor = pollard_rho(1001, max_iterations=10000)
        assert factor is not None
        assert 1001 % factor == 0
        assert 1 < factor < 1001

    def test_even_number_returns_two(self):
        assert pollard_rho(100) == 2

    def test_seed_reproducibility(self):
        # Same seed → same (deterministic) result
        f1 = pollard_rho(1001, max_iterations=1000, seed=42)
        f2 = pollard_rho(1001, max_iterations=1000, seed=42)
        assert f1 == f2


# ══════════════════════════════════════════════════════════════
# Individual benchmark wrappers
# ══════════════════════════════════════════════════════════════
class TestBenchmarkWrappers:

    def test_pollard_result_structure(self):
        result = benchmark_pollard_rho(15, timeout=5)
        assert isinstance(result, BenchmarkResult)
        assert result.algorithm == "pollard_rho"
        assert result.N == 15
        assert result.elapsed_sec >= 0

    def test_ecm_result_structure(self):
        result = benchmark_ecm(1001, timeout=5)
        assert isinstance(result, BenchmarkResult)
        assert result.algorithm == "ecm"
        assert result.N == 1001

    def test_fermat_result_structure(self):
        result = benchmark_fermat(143, timeout=5)
        assert isinstance(result, BenchmarkResult)
        assert result.algorithm == "fermat"
        assert result.N == 143


# ══════════════════════════════════════════════════════════════
# Master runner output schema
# ══════════════════════════════════════════════════════════════
class TestMasterRunner:

    def test_all_keys_present(self):
        result = benchmark_semiprime(15, timeout=5)
        required_keys = {
            "N",
            "best_algorithm", "min_runtime", "log_min_runtime",
            "runtime_pollard_rho", "success_pollard_rho", "timeout_pollard_rho",
            "runtime_ecm", "success_ecm", "timeout_ecm",
            "runtime_pollard_pm1", "success_pollard_pm1", "timeout_pollard_pm1",
            "runtime_fermat", "success_fermat", "timeout_fermat",
        }
        assert required_keys.issubset(result.keys())

    def test_types(self):
        result = benchmark_semiprime(15, timeout=5)
        assert isinstance(result["N"], int)
        assert isinstance(result["best_algorithm"], str)
        assert isinstance(result["min_runtime"], float)
        assert isinstance(result["log_min_runtime"], float)
        for algo in ["pollard_rho", "pollard_pm1", "ecm", "fermat"]:
            assert isinstance(result[f"runtime_{algo}"], float)
            assert isinstance(result[f"success_{algo}"], bool)
            assert isinstance(result[f"timeout_{algo}"], bool)

    def test_algorithm_selection_consistency(self):
        """best_algorithm must match the algorithm with minimum successful runtime."""
        result = benchmark_semiprime(77, timeout=5)

        if result["best_algorithm"] == "none":
            # No algorithm succeeded — acceptable for very small or edge cases
            assert result["min_runtime"] == -1.0
        else:
            # best_algorithm should have the lowest runtime among successful ones
            algo = result["best_algorithm"]
            assert result[f"success_{algo}"] is True
            best_time = result[f"runtime_{algo}"]
            for other in ["pollard_rho", "pollard_pm1", "ecm", "fermat"]:
                if result[f"success_{other}"]:
                    assert result[f"runtime_{other}"] >= best_time - 1e-9

    def test_algorithm_map_completeness(self):
        """Every algorithm in ALGORITHM_MAP must have a callable function."""
        assert "pollard_rho" in ALGORITHM_MAP
        assert "ecm" in ALGORITHM_MAP
        assert "pollard_pm1" in ALGORITHM_MAP
        assert "fermat" in ALGORITHM_MAP
        for name, fn in ALGORITHM_MAP.items():
            assert callable(fn)


# ══════════════════════════════════════════════════════════════
# Batch runner
# ══════════════════════════════════════════════════════════════
class TestBatchRunner:

    def test_batch_output_count(self):
        N_list = [15, 35, 77, 143, 221]
        results = benchmark_batch(N_list, timeout=5, batch_size=2, verbose=False)
        assert len(results) == len(N_list)
        for i, r in enumerate(results):
            assert r["N"] == N_list[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
