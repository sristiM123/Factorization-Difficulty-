# RSA Hardness Prediction & PQC Migration Framework

A hybrid machine learning framework for predicting RSA semiprime factorization
difficulty from the public modulus alone, with algorithm selection guidance and
NIST-grounded post-quantum migration scoring.

## Authors

- Sristi Mitra — Ss. Cyril and Methodius University, Skopje
- Zishnendu Sarker — University of Dhaka

## Overview

This framework predicts three things for any RSA semiprime `N = p × q`:

1. **Difficulty estimate** — how long factorization will take (regression)
2. **Best algorithm** — which of Pollard's rho, ECM, or Quadratic Sieve is fastest (classification)
3. **PQC migration urgency** — NIST-grounded recommendation for transitioning to post-quantum schemes

All predictions are computed from the public modulus `N` alone, requiring no knowledge of `p` or `q`.

## Key Contributions

- **First ML framework** predicting actual factorization runtime (not synthetic scores) from public modulus features
- **Rice's Algorithm Selection Framework** applied to cryptographic factorization for the first time
- **SHAP-based per-algorithm interpretability** revealing which features matter for each factoring strategy
- **Novel feature** — Fermat proximity `ceil(√N)² − N` for factor balance prediction
- **NIST-grounded PQC migration scoring** connecting empirical predictions to SP 800-57 Rev. 5 standards

## Project Structure
RSA-Hardness-Framework/
│
├── config/                    # Centralized configuration
│   └── settings.py            # Seeds, paths, hyperparameters, PQC tables
│
├── src/                       # Reusable source modules
│   ├── generators.py          # Prime sieve, semiprime generation
│   ├── features.py            # 37 features from N only
│   ├── benchmarks.py          # Pollard rho, ECM, QS timing
│   ├── models.py              # XGBoost, CatBoost, DualBranch MLP, ElasticNet
│   ├── evaluation.py          # SHAP, Bland-Altman, bootstrap, ablation
│   ├── pqc.py                 # NIST mapping, migration scoring
│   ├── visualization.py       # Publication-quality plotting
│   └── utils.py               # Reproducibility, I/O, logging
│
├── tests/                     # Unit tests
├── scripts/                   # Command-line entry points
├── notebooks/                 # Phase-wise analysis notebooks
│   ├── 01_dataset_generation.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_validation.ipynb
│   ├── 05_pqc_migration.ipynb
│   └── 06_paper_figures.ipynb
│
├── data/                      # Data artifacts (gitignored)
│   ├── raw/                   # Generated semiprimes + hidden factors
│   ├── processed/             # Final dataset with features + runtimes
│   └── meta/                  # JSON metadata, checksums
│
├── models/                    # Serialized trained models (gitignored)
├── results/                   # Experimental outputs
│   ├── figures/               # Publication-quality plots
│   ├── tables/                # CSV/LaTeX result tables
│   └── logs/                  # Training logs
│
├── docs/                      # Documentation and paper drafts
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md

## Installation

### Hardware Requirements

- **Minimum:** 8 GB RAM, 4-core CPU, 10 GB free disk
- **Recommended:** 16 GB RAM for faster training
- **GPU:** Optional

### Software

Python 3.10, 3.11, 3.12, or 3.13.

### Setup in IntelliJ IDEA / PyCharm

1. Clone or download the project
2. `File → Open` and select the `RSA-Hardness-Framework` folder
3. Configure Python interpreter: `File → Project Structure → Project → SDK`
4. Mark directories:
    - `src/`, `config/` → Sources Root (blue)
    - `tests/` → Test Sources Root (green)
    - `data/`, `models/`, `results/` → Excluded (orange)
5. Install dependencies:
```bash
   pip install -r requirements.txt
```

### Setup via Command Line

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

## Running the Pipeline

### Memory-Safe Mode (for 8 GB RAM systems)

In `config/settings.py`, set:

```python
PROFILE = "memory_safe"
```

This uses 50,000 samples, chunked processing, and keeps RAM under ~4 GB.

### Full Mode (for 16 GB RAM systems)

```python
PROFILE = "full"
```

Uses the complete 100,000-sample dataset.

### Execution Order

Run notebooks sequentially:
01_dataset_generation.ipynb   →  generates semiprimes + benchmarks (2–4 hours)
02_feature_analysis.ipynb     →  feature analysis, leakage audit
03_model_training.ipynb       →  trains hybrid ensemble
04_validation.ipynb           →  SHAP, Bland-Altman, bootstrap, ablation
05_pqc_migration.ipynb        →  NIST mapping, migration scoring
06_paper_figures.ipynb        →  publication-quality figures

## Reproducibility

- Global seed: `SEED = 42` (in `config/settings.py`)
- Dataset checksum stored in `data/meta/dataset_metadata.json`
- Environment snapshot captured in every run
- All random operations consistently seeded

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{mitra2025rsa,
  title={A Hybrid Learning Framework for RSA Vulnerability Assessment and
         Policy-Oriented Post-Quantum Migration Guidance},
  author={Mitra, Sristi and Sarker, Zishnendu},
  journal={Expert Systems with Applications},
  year={2025},
  note={Under review}
}
```

## License

MIT License — see LICENSE file for details.

## Contact

For questions, issues, or collaboration inquiries, please contact the authors
or open an issue on the repository.