"""
generators.py — Prime sieving and semiprime generation.

Memory-efficient implementation for 8GB RAM systems.
Prime pool built in chunks, semiprimes generated via controlled sampling.

The generated semiprimes span 32–160 bits with controlled factor-structure
diversity (balanced, moderate, unbalanced, special) to ensure the algorithm-
selection task is meaningful.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ══════════════════════════════════════════════════════════════
# Data class
# ══════════════════════════════════════════════════════════════
@dataclass
class SemiprimeRecord:
    """A single semiprime with generation metadata.

    Attributes
    ----------
    N : int
        The public modulus (p * q).
    bit_length : int
        Bit length of N.
    p, q : int
        Prime factors. HIDDEN — used only for validation, never as features.
    structure : str
        "balanced", "moderate", "unbalanced", or "special".
    p_bits, q_bits : int
        Bit lengths of the factors.
    balance_ratio : float
        min(p, q) / max(p, q) — closer to 1.0 means more balanced.
    """
    N: int
    bit_length: int
    p: int
    q: int
    structure: str
    p_bits: int
    q_bits: int
    balance_ratio: float


# ══════════════════════════════════════════════════════════════
# Prime sieve
# ══════════════════════════════════════════════════════════════
def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Return all primes up to `limit` (inclusive). Uses a memory-efficient bytearray."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = bytearray(
                len(range(p * p, limit + 1, p))
            )
    return [i for i, v in enumerate(is_prime) if v]


def generate_prime_pool(
        max_bits: int,
        primes_per_size: int = 2000,
) -> Dict[int, List[int]]:
    """
    Build {bit_length: [primes]} dictionary.

    - Small primes (≤24 bits) generated via Sieve of Eratosthenes.
    - Larger primes generated via sympy.randprime() for uniform sampling.

    Parameters
    ----------
    max_bits : int
        Maximum prime bit-length needed.
    primes_per_size : int
        Target number of primes per bit-length (lower → less memory).

    Returns
    -------
    dict[int, list[int]]
        Mapping from bit-length to list of primes with that exact bit-length.
    """
    pool: Dict[int, List[int]] = {}

    # Small primes via sieve
    sieve_limit = min(2 ** 24, 2 ** max_bits)
    for p in sieve_of_eratosthenes(sieve_limit):
        pool.setdefault(p.bit_length(), []).append(p)

    # Larger primes via sympy
    try:
        from sympy import randprime
        for bits in range(25, max_bits + 1):
            existing = pool.get(bits, [])
            if len(existing) >= primes_per_size:
                continue
            lo, hi = 2 ** (bits - 1), 2 ** bits - 1
            attempts = 0
            while len(existing) < primes_per_size and attempts < 20_000:
                cand = randprime(lo, hi + 1)
                if cand.bit_length() == bits:
                    existing.append(cand)
                attempts += 1
            pool[bits] = existing
    except ImportError:
        # Sieve-only pool if sympy unavailable
        pass

    return pool


# ══════════════════════════════════════════════════════════════
# Factor-structure assignment
# ══════════════════════════════════════════════════════════════
def _pick_factor_bits(
        total_bits: int,
        structure: str,
        rng: random.Random,
) -> Tuple[int, int]:
    """Pick (p_bits, q_bits) given target total bit-length and structure type."""
    half = total_bits // 2

    if structure == "balanced":
        # p ≈ q (within a few bits)
        p_bits = half + rng.randint(-1, 1)
        q_bits = total_bits - p_bits + rng.choice([-1, 0, 1])

    elif structure == "moderate":
        # One factor 60–80% bit-length of the other
        ratio = rng.uniform(0.60, 0.80)
        p_bits = max(10, int(half * ratio))
        q_bits = total_bits - p_bits + rng.choice([-1, 0, 1])

    elif structure == "unbalanced":
        # One factor < 50% bit-length of the other
        ratio = rng.uniform(0.25, 0.50)
        p_bits = max(8, int(half * ratio))
        q_bits = total_bits - p_bits + rng.choice([-1, 0, 1])

    elif structure == "special":
        # Near-equal factors (Fermat-vulnerable)
        p_bits = half
        q_bits = half + rng.choice([0, 1])

    else:
        raise ValueError(f"Unknown structure: {structure!r}")

    return max(8, p_bits), max(8, q_bits)


# ══════════════════════════════════════════════════════════════
# Semiprime generation
# ══════════════════════════════════════════════════════════════
def generate_semiprimes(
        bit_ranges: Dict[Tuple[int, int], int],
        structure_ratios: Dict[str, float],
        prime_pool: Dict[int, List[int]],
        seed: int = 42,
        verbose: bool = True,
) -> List[SemiprimeRecord]:
    """
    Generate semiprimes with controlled factor-structure diversity.

    Parameters
    ----------
    bit_ranges : dict
        {(min_bits, max_bits): count_of_semiprimes}
    structure_ratios : dict
        {"balanced": 0.40, "moderate": 0.30, ...}
    prime_pool : dict
        {bit_length: [primes]} from generate_prime_pool().
    seed : int
        Reproducibility seed.
    verbose : bool
        Print progress messages.

    Returns
    -------
    list[SemiprimeRecord]
        Generated semiprimes with metadata.
    """
    rng = random.Random(seed)
    records: List[SemiprimeRecord] = []

    for (lo_bits, hi_bits), count in bit_ranges.items():
        if verbose:
            print(f"\n  Generating {count:,} semiprimes in "
                  f"[{lo_bits}, {hi_bits}] bits...")

        # Allocate counts across structure types
        struct_counts = {
            s: int(count * r) for s, r in structure_ratios.items()
        }
        # Add any leftover (from integer rounding) to "balanced"
        struct_counts["balanced"] += count - sum(struct_counts.values())

        for struct, n_needed in struct_counts.items():
            generated = 0
            attempts = 0
            max_attempts = n_needed * 50

            while generated < n_needed and attempts < max_attempts:
                attempts += 1
                target_bits = rng.randint(lo_bits, hi_bits)
                p_target, q_target = _pick_factor_bits(target_bits, struct, rng)

                p_primes = prime_pool.get(p_target)
                q_primes = prime_pool.get(q_target)
                if not p_primes or not q_primes:
                    continue

                p = rng.choice(p_primes)
                q = rng.choice(q_primes)
                if p == q:
                    continue

                N = p * q
                bl = N.bit_length()
                if lo_bits <= bl <= hi_bits:
                    records.append(SemiprimeRecord(
                        N=N,
                        bit_length=bl,
                        p=min(p, q),
                        q=max(p, q),
                        structure=struct,
                        p_bits=min(p, q).bit_length(),
                        q_bits=max(p, q).bit_length(),
                        balance_ratio=min(p, q) / max(p, q),
                    ))
                    generated += 1

            if verbose:
                print(f"    {struct:>12s}: {generated:>6,} / {n_needed:,}")

    rng.shuffle(records)
    if verbose:
        print(f"\n  Total generated: {len(records):,}")
    return records