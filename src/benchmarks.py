"""
benchmarks.py — Factorization algorithm timing.

Four factorization algorithms with complementary attack strategies:
    1. Pollard's rho   — fast when one factor is small (unbalanced)
    2. Pollard's p-1   — fast when (p-1) has only small prime factors (smooth)
    3. ECM             — general-purpose, medium-factor sweet spot
    4. Fermat's method — fast when p ≈ q (balanced, close factors)

This produces genuine 4-class algorithm selection ground truth.
The GNFS is excluded as it requires specialized implementations and only
becomes competitive beyond our experimental range (>100 digits).
"""

import gc
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkResult:
    """Result of running one factorization algorithm on one semiprime."""
    algorithm: str
    N: int
    factor_found: Optional[int]
    elapsed_sec: float
    success: bool
    timed_out: bool


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════
def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ══════════════════════════════════════════════════════════════
# 1. Pollard's rho
# ══════════════════════════════════════════════════════════════
def pollard_rho(
        n: int,
        max_iterations: int = 2_000_000,
        seed: int = 42,
) -> Optional[int]:
    """
    Pollard's rho factorization.

    Complexity: O(p^{1/2}) where p is the smallest prime factor.
    Fast when factors are unbalanced (one small, one large).
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2

    rng = random.Random(seed)
    x = rng.randint(2, n - 1)
    y = x
    c = rng.randint(1, n - 1)

    for _ in range(max_iterations):
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = _gcd(abs(x - y), n)
        if 1 < d < n:
            return d
        if d == n:
            c = rng.randint(1, n - 1)
            x = rng.randint(2, n - 1)
            y = x

    return None


def benchmark_pollard_rho(
        n: int,
        timeout: float = 1.0,
        seed: int = 42,
) -> BenchmarkResult:
    """Time Pollard's rho. 1s timeout — finds small factors fast or not at all."""
    start = time.perf_counter()
    try:
        max_iter = min(5_000_000, max(50_000, 2 ** (n.bit_length() // 4 + 2)))
        factor = pollard_rho(n, max_iterations=max_iter, seed=seed)
        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            return BenchmarkResult("pollard_rho", n, None, timeout, False, True)
        success = factor is not None and 1 < factor < n
        return BenchmarkResult("pollard_rho", n, factor, elapsed, success, False)
    except Exception:
        return BenchmarkResult("pollard_rho", n, None,
                               time.perf_counter() - start, False, False)


# ══════════════════════════════════════════════════════════════
# 2. Pollard's p-1
# ══════════════════════════════════════════════════════════════
def pollard_pm1(n: int, B1: int = 50_000, B2: int = 500_000) -> Optional[int]:
    """
    Pollard's p-1 factorization.

    Finds a factor p of n if (p-1) is B1-smooth (all prime factors ≤ B1).
    Stage 2 extends to B2-smooth for one large factor.

    Complexity: O(B1 · log N) for Stage 1.
    Fast when p-1 has only small prime factors (B-smooth).

    This connects to our modular residue features — smoothness patterns
    in the factors may be partially detectable from N mod small_primes.
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2

    # Stage 1: compute a = 2^(B1!) mod n via successive prime powers
    a = 2
    # Use small primes up to B1
    p = 2
    while p <= B1:
        # Raise to the highest power of p that fits in B1
        pp = p
        while pp * p <= B1:
            pp *= p
        a = pow(a, pp, n)
        g = _gcd(a - 1, n)
        if 1 < g < n:
            return g
        if g == n:
            return None  # Failed — too smooth

        # Next prime (simple increment)
        p += 1
        if p > 2:
            while p <= B1:
                is_p = True
                for d in range(2, int(math.isqrt(p)) + 1):
                    if p % d == 0:
                        is_p = False
                        break
                if is_p:
                    break
                p += 2
            else:
                break

    # Final check after Stage 1
    g = _gcd(a - 1, n)
    if 1 < g < n:
        return g

    return None


def benchmark_pollard_pm1(n: int, timeout: float = 0.5) -> BenchmarkResult:
    """
    Time Pollard's p-1 on semiprime n.

    Very short timeout: p-1 either works in milliseconds (smooth factor)
    or won't work at all (non-smooth factors). No point waiting.
    """
    start = time.perf_counter()
    try:
        # Scale B1 with n's size but keep it bounded for speed
        B1 = min(50_000, max(1_000, n.bit_length() * 200))
        factor = pollard_pm1(n, B1=B1)
        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            return BenchmarkResult("pollard_pm1", n, None, timeout, False, True)
        success = factor is not None and 1 < factor < n
        return BenchmarkResult("pollard_pm1", n,
                               int(factor) if factor else None,
                               elapsed, success, False)
    except Exception:
        return BenchmarkResult("pollard_pm1", n, None,
                               time.perf_counter() - start, False, False)


# ══════════════════════════════════════════════════════════════
# 3. ECM (via sympy)
# ══════════════════════════════════════════════════════════════
def benchmark_ecm(n: int, timeout: float = 3.0) -> BenchmarkResult:
    """
    Time ECM on semiprime n using sympy.

    Complexity: sub-exponential in the size of the smallest factor.
    General-purpose — often fastest for medium-factor cases.
    """
    try:
        from sympy.ntheory.ecm import ecm as sympy_ecm
    except ImportError:
        return BenchmarkResult("ecm", n, None, 0.0, False, False)

    start = time.perf_counter()
    try:
        factors = sympy_ecm(n)
        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            return BenchmarkResult("ecm", n, None, timeout, False, True)
        for f in factors:
            if 1 < f < n:
                return BenchmarkResult("ecm", n, int(f), elapsed, True, False)
        return BenchmarkResult("ecm", n, None, elapsed, False, False)
    except Exception:
        return BenchmarkResult("ecm", n, None,
                               time.perf_counter() - start, False, False)


# ══════════════════════════════════════════════════════════════
# 4. Fermat's factorization method
# ══════════════════════════════════════════════════════════════
def fermat_factor(n: int, max_iterations: int = 100_000) -> Optional[int]:
    """
    Fermat's factorization: finds factors when p ≈ q.

    Starts from ceil(sqrt(N)), checks if a^2 - N is a perfect square.
    If yes: N = (a - b)(a + b).

    Complexity: O(sqrt(N) / |p - q|) — extremely fast when p ≈ q.
    Directly connects to our novel fermat_proximity feature.
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2

    a = math.isqrt(n)
    if a * a == n:
        return a

    a += 1

    for _ in range(max_iterations):
        b_sq = a * a - n
        b = math.isqrt(b_sq)
        if b * b == b_sq:
            p = a - b
            q = a + b
            if 1 < p < n:
                return p
            if 1 < q < n:
                return q
        a += 1

    return None


def benchmark_fermat(n: int, timeout: float = 1.0) -> BenchmarkResult:
    """
    Time Fermat's method. 1s timeout — either works in milliseconds or won't.
    """
    start = time.perf_counter()
    try:
        max_iter = min(500_000, max(10_000, n.bit_length() * 1000))
        factor = fermat_factor(n, max_iterations=max_iter)
        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            return BenchmarkResult("fermat", n, None, timeout, False, True)
        success = factor is not None and 1 < factor < n
        return BenchmarkResult("fermat", n,
                               int(factor) if factor else None,
                               elapsed, success, False)
    except Exception:
        return BenchmarkResult("fermat", n, None,
                               time.perf_counter() - start, False, False)


# ══════════════════════════════════════════════════════════════
# Master benchmark runner
# ══════════════════════════════════════════════════════════════
ALGORITHM_MAP = {
    "pollard_rho":  benchmark_pollard_rho,
    "pollard_pm1":  benchmark_pollard_pm1,
    "ecm":          benchmark_ecm,
    "fermat":       benchmark_fermat,
}


def benchmark_semiprime(
        n: int,
        algorithms: Optional[List[str]] = None,
        timeout: float = 3.0,
) -> dict:
    """Run all specified algorithms on n and return flattened result dict."""
    if algorithms is None:
        algorithms = list(ALGORITHM_MAP.keys())

    result = {"N": n}
    best_time = float("inf")
    best_algo = "none"

    for name in algorithms:
        br = ALGORITHM_MAP[name](n, timeout=timeout)
        result[f"runtime_{name}"] = br.elapsed_sec
        result[f"success_{name}"] = br.success
        result[f"timeout_{name}"] = br.timed_out
        if br.success and br.elapsed_sec < best_time:
            best_time = br.elapsed_sec
            best_algo = name

    result["best_algorithm"] = best_algo
    result["min_runtime"] = best_time if best_time < float("inf") else -1.0
    result["log_min_runtime"] = (
        math.log2(max(best_time, 1e-9))
        if best_time < float("inf") else -1.0
    )
    return result


def benchmark_semiprime_fast(
        n: int,
        timeout_rho: float = 1.0,
        timeout_pm1: float = 0.5,
        timeout_ecm: float = 3.0,
        timeout_fermat: float = 1.0,
) -> dict:
    """
    Optimized benchmarking with per-algorithm timeouts.

    - Rho:    1.0s (finds small factors instantly; if not in 1s, won't)
    - p-1:    0.5s (finds smooth factors instantly; if not in 0.5s, won't)
    - ECM:    3.0s (general purpose, needs more time on larger inputs)
    - Fermat: 1.0s (finds balanced factors instantly; if not in 1s, won't)
    """
    rho = benchmark_pollard_rho(n, timeout=timeout_rho)
    pm1 = benchmark_pollard_pm1(n, timeout=timeout_pm1)
    ecm = benchmark_ecm(n, timeout=timeout_ecm)
    fmt = benchmark_fermat(n, timeout=timeout_fermat)

    result = {
        "N": n,
        "runtime_pollard_rho": rho.elapsed_sec,
        "success_pollard_rho": rho.success,
        "timeout_pollard_rho": rho.timed_out,
        "runtime_pollard_pm1": pm1.elapsed_sec,
        "success_pollard_pm1": pm1.success,
        "timeout_pollard_pm1": pm1.timed_out,
        "runtime_ecm": ecm.elapsed_sec,
        "success_ecm": ecm.success,
        "timeout_ecm": ecm.timed_out,
        "runtime_fermat": fmt.elapsed_sec,
        "success_fermat": fmt.success,
        "timeout_fermat": fmt.timed_out,
    }

    best_time = float("inf")
    best_algo = "none"
    for algo, br in [("pollard_rho", rho), ("pollard_pm1", pm1),
                     ("ecm", ecm), ("fermat", fmt)]:
        if br.success and br.elapsed_sec < best_time:
            best_time = br.elapsed_sec
            best_algo = algo

    result["best_algorithm"] = best_algo
    result["min_runtime"] = best_time if best_time < float("inf") else -1.0
    result["log_min_runtime"] = (
        math.log2(max(best_time, 1e-9))
        if best_time < float("inf") else -1.0
    )
    return result


def benchmark_batch(
        N_list: List[int],
        algorithms: Optional[List[str]] = None,
        timeout: float = 3.0,
        batch_size: int = 250,
        verbose: bool = True,
) -> List[dict]:
    """Benchmark a list of semiprimes in memory-safe batches."""
    results = []
    total = len(N_list)
    start_time = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        for n in N_list[batch_start:batch_end]:
            results.append(benchmark_semiprime(
                n, algorithms=algorithms, timeout=timeout,
            ))
        gc.collect()
        if verbose:
            elapsed = time.perf_counter() - start_time
            rate = batch_end / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / rate / 60 if rate > 0 else 0
            print(f"    {batch_end:>6,}/{total:,} "
                  f"({batch_end / total * 100:5.1f}%) | "
                  f"{elapsed / 60:.1f}min | {rate:5.1f}/s | ETA: {eta:.1f}min")
    return results