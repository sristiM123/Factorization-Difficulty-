"""Tests for src/features.py — verifies all 39 features behave correctly."""

import pytest
from src.features import (
    hamming_weight, binary_entropy, byte_entropy,
    longest_run, bit_transition_count,
    fermat_proximity, sqrt_fractional_part,
    digital_root, jacobi_symbol,
    extract_features, get_feature_names,
)


# ══════════════════════════════════════════════════════════════
# Basic feature computations
# ══════════════════════════════════════════════════════════════
class TestBasicFeatures:

    def test_hamming_weight(self):
        assert hamming_weight(0) == 0
        assert hamming_weight(1) == 1
        assert hamming_weight(7) == 3       # 0b111
        assert hamming_weight(15) == 4      # 0b1111
        assert hamming_weight(256) == 1     # 0b100000000

    def test_binary_entropy_bounds(self):
        # All 1s → 0 entropy (no uncertainty)
        assert binary_entropy(15) == 0.0    # 0b1111
        # Balanced bits → entropy 1.0 (maximum)
        assert binary_entropy(10) == 1.0    # 0b1010
        # General values must be in [0, 1]
        for n in [1, 7, 100, 1000, 2**20 - 1]:
            ent = binary_entropy(n)
            assert 0.0 <= ent <= 1.0

    def test_byte_entropy_nonneg(self):
        for n in [1, 256, 65535, 2 ** 32, 2 ** 64]:
            assert byte_entropy(n) >= 0

    def test_longest_run(self):
        # 0b11100011 → longest run of 1s is 3, of 0s is 3
        assert longest_run(0b11100011, "1") == 3
        assert longest_run(0b11100011, "0") == 3
        assert longest_run(0b1111, "1") == 4
        assert longest_run(0b10101, "1") == 1

    def test_bit_transitions(self):
        # 0b1010 has 3 transitions: 1→0, 0→1, 1→0
        assert bit_transition_count(0b1010) == 3
        # 0b1111 has 0 transitions
        assert bit_transition_count(0b1111) == 0
        # 0b11110000 has 1 transition
        assert bit_transition_count(0b11110000) == 1


# ══════════════════════════════════════════════════════════════
# Fermat / structural features
# ══════════════════════════════════════════════════════════════
class TestFermatFeatures:

    def test_fermat_proximity_perfect_square(self):
        # N = 100 → sqrt = 10 exactly → proximity = 0
        assert fermat_proximity(100) == 0
        assert fermat_proximity(144) == 0

    def test_fermat_proximity_positive(self):
        # N = 15, isqrt = 3, ceil = 4, 16 - 15 = 1
        assert fermat_proximity(15) == 1
        # N = 35 (5×7), isqrt = 5, ceil = 6, 36 - 35 = 1
        assert fermat_proximity(35) == 1

    def test_fermat_proximity_small_for_balanced(self):
        # Balanced factors → small fermat_proximity
        # 143 = 11×13 (close), should be small
        # 221 = 13×17 (close), should be small
        assert fermat_proximity(143) < 50
        assert fermat_proximity(221) < 100

    def test_sqrt_fractional_bounds(self):
        for n in [2, 7, 15, 1000, 123456]:
            frac = sqrt_fractional_part(n)
            assert 0.0 <= frac < 1.0


# ══════════════════════════════════════════════════════════════
# Number-theoretic features
# ══════════════════════════════════════════════════════════════
class TestNumberTheory:

    def test_digital_root(self):
        assert digital_root(0) == 0
        assert digital_root(1) == 1
        assert digital_root(38) == 2   # 3+8=11 → 1+1=2
        assert digital_root(999) == 9  # 9+9+9=27 → 2+7=9
        assert digital_root(12345) == 6  # 1+2+3+4+5=15 → 1+5=6

    def test_jacobi_symbol(self):
        # Known values
        assert jacobi_symbol(2, 15) == 1
        assert jacobi_symbol(7, 15) == -1
        # n must be odd positive
        assert jacobi_symbol(5, 10) == 0   # n even
        assert jacobi_symbol(5, 0) == 0    # n zero


# ══════════════════════════════════════════════════════════════
# Feature extraction (integration)
# ══════════════════════════════════════════════════════════════
class TestFeatureExtraction:

    def test_extract_features_shape(self):
        N_values = [15, 35, 77, 143, 221, 323]
        df = extract_features(N_values, verbose=False)
        assert len(df) == len(N_values)
        # Feature count must be 39 per the documented design
        assert df.shape[1] == 39

    def test_feature_names_count(self):
        names = get_feature_names()
        assert len(names) == 39

    def test_feature_names_includes_novel_features(self):
        names = get_feature_names()
        # Novel contributions must be present
        assert "fermat_proximity" in names
        assert "log_fermat_prox" in names
        assert "jacobi_symbol_sum" in names
        assert "mod_residue_variance" in names

    def test_no_nan(self):
        N_values = [15, 77, 1000003, 2 ** 20 - 1, 2 ** 30 + 3]
        df = extract_features(N_values, verbose=False)
        assert not df.isna().any().any(), "NaN values found in extracted features"

    def test_feature_column_consistency(self):
        """Every feature in get_feature_names() must appear in extract_features()."""
        N_values = [15, 35, 77]
        df = extract_features(N_values, verbose=False)
        for name in get_feature_names():
            assert name in df.columns, f"Missing feature: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])