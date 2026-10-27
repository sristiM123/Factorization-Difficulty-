"""
features.py — Number-theoretic feature engineering from N only.

All features are computed from the public modulus N alone.
No feature uses knowledge of p or q. This is the attacker-observable set.

Feature groups (37 total):
    1. Size (2):              log2_N, bit_length
    2. Binary structure (6):  hamming_weight, hw_ratio, binary_entropy,
                              longest_run_1s, longest_run_0s, bit_transitions
    3. Byte structure (1):    byte_entropy
    4. Modular residues (24): N mod {3, 5, 7, 11, 13, 17, 19, 23, 29}
                              plus N mod {256, 512, 1024}, both raw and fractional
    5. Fermat/structural (3): fermat_proximity, log_fermat_prox, sqrt_fraction
    6. Number-theoretic (3):  digital_root, mod_residue_variance, jacobi_symbol_sum

Novel contribution: `fermat_proximity` — never used before as an ML feature.

Memory note: `extract_features()` processes in chunks for 8GB RAM systems.
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# Individual feature computations
# ══════════════════════════════════════════════════════════════
def _binary_repr(n: int) -> str:
    """Binary string without the '0b' prefix."""
    return bin(n)[2:]


def hamming_weight(n: int) -> int:
    """Count of 1-bits in binary representation."""
    return bin(n).count("1")


def binary_entropy(n: int) -> float:
    """Shannon entropy of the binary digit distribution."""
    b = _binary_repr(n)
    if not b:
        return 0.0
    p1 = b.count("1") / len(b)
    p0 = 1.0 - p1
    ent = 0.0
    if p0 > 0:
        ent -= p0 * math.log2(p0)
    if p1 > 0:
        ent -= p1 * math.log2(p1)
    return ent


def byte_entropy(n: int) -> float:
    """Shannon entropy over byte-value distribution of n."""
    n_bytes = max(1, (n.bit_length() + 7) // 8)
    byte_vals = [(n >> (8 * i)) & 0xFF for i in range(n_bytes)]
    counts: dict = {}
    for b in byte_vals:
        counts[b] = counts.get(b, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / n_bytes
        ent -= p * math.log2(p)
    return ent


def longest_run(n: int, bit: str = "1") -> int:
    """Longest consecutive run of a specified bit ('0' or '1') in binary n."""
    b = _binary_repr(n)
    max_run = cur = 0
    for ch in b:
        if ch == bit:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def bit_transition_count(n: int) -> int:
    """Number of 0↔1 transitions in the binary representation."""
    b = _binary_repr(n)
    return sum(1 for i in range(1, len(b)) if b[i] != b[i - 1])


def fermat_proximity(n: int) -> int:
    """
    Fermat proximity: ceil(sqrt(N))^2 - N.

    Small values indicate p ≈ q, making Fermat factorization fast.
    Key novel feature for predicting factor balance from N alone.
    """
    s = math.isqrt(n)
    if s * s == n:
        return 0
    s += 1
    return s * s - n


def sqrt_fractional_part(n: int) -> float:
    """Fractional part of sqrt(N). Correlates weakly with factor balance."""
    sq = math.sqrt(float(n))
    return sq - math.floor(sq)


def digital_root(n: int) -> int:
    """Repeated digit sum until a single digit. A number-theoretic invariant."""
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def jacobi_symbol(a: int, n: int) -> int:
    """
    Compute Jacobi symbol (a/n) for odd positive n.

    Returns 0 if gcd(a, n) > 1, otherwise ±1.
    Captures quadratic-residue structure that relates to factorization methods.
    """
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


# ══════════════════════════════════════════════════════════════
# Canonical feature ordering (used across training and inference)
# ══════════════════════════════════════════════════════════════
def get_feature_names(
        small_primes: Optional[List[int]] = None,
        power_mods: Optional[List[int]] = None,
) -> List[str]:
    """Return the canonical ordered list of feature names used by the model."""
    small_primes = small_primes or [3, 5, 7, 11, 13, 17, 19, 23, 29]
    power_mods = power_mods or [256, 512, 1024]

    names = [
        # Size
        "log2_N", "bit_length",
        # Binary structure
        "hamming_weight", "hw_ratio", "binary_entropy",
        "longest_run_1s", "longest_run_0s", "bit_transitions",
        # Byte structure
        "byte_entropy",
        # Fermat / structural
        "fermat_proximity", "log_fermat_prox", "sqrt_fraction",
        # Number-theoretic
        "digital_root",
    ]
    for sp in small_primes:
        names += [f"N_mod_{sp}", f"N_mod_{sp}_frac"]
    for pm in power_mods:
        names += [f"N_mod_{pm}", f"N_mod_{pm}_frac"]
    names += ["mod_residue_variance", "jacobi_symbol_sum"]
    return names


# ══════════════════════════════════════════════════════════════
# Extraction — one sample and batch
# ══════════════════════════════════════════════════════════════
def _extract_one(
        N: int,
        small_primes: List[int],
        power_mods: List[int],
        jacobi_bases: List[int],
) -> dict:
    """Compute all features for a single N."""
    bl = N.bit_length()
    hw = hamming_weight(N)
    fp = fermat_proximity(N)

    feat = {
        "log2_N":          math.log2(N) if N > 0 else 0.0,
        "bit_length":      bl,
        "hamming_weight":  hw,
        "hw_ratio":        hw / bl if bl > 0 else 0.0,
        "binary_entropy":  binary_entropy(N),
        "longest_run_1s":  longest_run(N, "1"),
        "longest_run_0s":  longest_run(N, "0"),
        "bit_transitions": bit_transition_count(N),
        "byte_entropy":    byte_entropy(N),
        "fermat_proximity": fp,
        "log_fermat_prox": math.log2(fp + 1),
        "sqrt_fraction":   sqrt_fractional_part(N),
        "digital_root":    digital_root(N),
    }

    # Modular residues (small primes)
    residues = []
    for sp in small_primes:
        r = N % sp
        feat[f"N_mod_{sp}"] = r
        feat[f"N_mod_{sp}_frac"] = r / sp
        residues.append(r / sp)

    # Modular residues (powers of 2)
    for pm in power_mods:
        r = N % pm
        feat[f"N_mod_{pm}"] = r
        feat[f"N_mod_{pm}_frac"] = r / pm

    feat["mod_residue_variance"] = float(np.var(residues))

    # Jacobi symbols (only defined for odd n)
    jac_sum = 0
    if N % 2 == 1:
        for base in jacobi_bases:
            jac_sum += jacobi_symbol(base, N)
    feat["jacobi_symbol_sum"] = jac_sum

    return feat


def extract_features(
        N_values: List[int],
        small_primes: Optional[List[int]] = None,
        power_mods: Optional[List[int]] = None,
        jacobi_bases: Optional[List[int]] = None,
        chunk_size: int = 10_000,
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract all features from a list of N values, in memory-safe chunks.

    Parameters
    ----------
    N_values : list[int]
        Semiprimes (public moduli).
    small_primes, power_mods, jacobi_bases : list[int]
        Feature parameters. Use defaults unless changing the feature set.
    chunk_size : int
        Process in chunks of this size. Lower → less RAM.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        One row per N, columns = feature names.
    """
    small_primes = small_primes or [3, 5, 7, 11, 13, 17, 19, 23, 29]
    power_mods = power_mods or [256, 512, 1024]
    jacobi_bases = jacobi_bases or [2, 3, 5, 7, 11]

    total = len(N_values)
    all_rows = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        for N in N_values[start:end]:
            all_rows.append(_extract_one(
                N, small_primes, power_mods, jacobi_bases,
            ))
        if verbose:
            print(f"    Features: {end:,} / {total:,}")

    df = pd.DataFrame(all_rows)
    if verbose:
        print(f"    Complete: {len(df)} samples x {len(df.columns)} features")
    return df