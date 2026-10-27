"""Tests for src/generators.py — verifies prime sieve and semiprime generation."""

import pytest
from src.generators import (
    sieve_of_eratosthenes,
    generate_prime_pool,
    generate_semiprimes,
    SemiprimeRecord,
)


# ══════════════════════════════════════════════════════════════
# Sieve of Eratosthenes
# ══════════════════════════════════════════════════════════════
class TestSieve:

    def test_small_primes(self):
        assert sieve_of_eratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_edge_cases(self):
        assert sieve_of_eratosthenes(0) == []
        assert sieve_of_eratosthenes(1) == []
        assert sieve_of_eratosthenes(2) == [2]

    def test_all_returned_are_prime(self):
        primes = sieve_of_eratosthenes(100)
        for p in primes:
            # Trial division: p is prime iff it has no divisors in [2, sqrt(p)]
            for i in range(2, int(p ** 0.5) + 1):
                assert p % i != 0, f"{p} returned by sieve but not prime"


# ══════════════════════════════════════════════════════════════
# Prime pool
# ══════════════════════════════════════════════════════════════
class TestPrimePool:

    def test_pool_structure(self):
        pool = generate_prime_pool(max_bits=10, primes_per_size=50)
        assert len(pool) > 0
        # Every prime must have the bit-length it's keyed under
        for bits, primes in pool.items():
            for p in primes:
                assert p.bit_length() == bits

    def test_pool_has_expected_ranges(self):
        pool = generate_prime_pool(max_bits=8)
        # Must contain at least some small-bit primes
        assert any(k <= 4 for k in pool.keys())


# ══════════════════════════════════════════════════════════════
# Semiprime generation
# ══════════════════════════════════════════════════════════════
class TestSemiprimeGeneration:

    def test_basic_generation(self):
        pool = generate_prime_pool(max_bits=20, primes_per_size=100)
        bit_ranges = {(16, 20): 50}
        structure_ratios = {
            "balanced": 0.50, "moderate": 0.30,
            "unbalanced": 0.15, "special": 0.05,
        }
        records = generate_semiprimes(
            bit_ranges=bit_ranges,
            structure_ratios=structure_ratios,
            prime_pool=pool,
            seed=42,
            verbose=False,
        )
        assert len(records) > 0
        for r in records:
            assert isinstance(r, SemiprimeRecord)
            assert 16 <= r.bit_length <= 20
            assert r.N == r.p * r.q       # p and q are genuine factors
            assert r.p <= r.q             # ordering invariant
            assert 0 < r.balance_ratio <= 1.0

    def test_reproducibility(self):
        """Same seed must produce identical output."""
        pool = generate_prime_pool(max_bits=16, primes_per_size=50)
        br = {(12, 16): 20}
        sr = {"balanced": 1.0, "moderate": 0.0,
              "unbalanced": 0.0, "special": 0.0}
        r1 = generate_semiprimes(br, sr, pool, seed=42, verbose=False)
        r2 = generate_semiprimes(br, sr, pool, seed=42, verbose=False)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.N == b.N
            assert a.structure == b.structure

    def test_structure_distribution(self):
        """Structure ratios should be roughly respected."""
        pool = generate_prime_pool(max_bits=18, primes_per_size=100)
        bit_ranges = {(14, 18): 1000}
        structure_ratios = {
            "balanced": 0.40, "moderate": 0.30,
            "unbalanced": 0.20, "special": 0.10,
        }
        records = generate_semiprimes(
            bit_ranges, structure_ratios, pool,
            seed=42, verbose=False,
        )
        struct_counts = {}
        for r in records:
            struct_counts[r.structure] = struct_counts.get(r.structure, 0) + 1
        # Each type should have roughly the expected ratio (within 30%)
        total = len(records)
        for s, expected_ratio in structure_ratios.items():
            actual = struct_counts.get(s, 0) / total
            assert abs(actual - expected_ratio) < 0.3, (
                f"Structure {s}: expected {expected_ratio:.2f}, got {actual:.2f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])