"""
RSA Hardness Prediction Framework — source package.

Public modules:
    generators    — Prime sieve, semiprime generation
    features      — Number-theoretic feature engineering
    benchmarks    — Factorization algorithm timing
    models        — XGBoost, CatBoost, DualBranch MLP, ElasticNet ensemble
    evaluation    — SHAP, Bland-Altman, bootstrap, ablation
    pqc           — NIST mapping, migration scoring
    visualization — Publication-quality plotting
    utils         — Reproducibility, I/O, logging
"""

__version__ = "1.0.0"
__author__ = "Sristi Mitra, Zishnendu Sarker"