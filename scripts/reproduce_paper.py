"""
reproduce_paper.py — Reproduce every number reported in the paper from the
released data.

Usage (from the repository root):
    python scripts/reproduce_paper.py            # all experiments
    python scripts/reproduce_paper.py --skip-fermat   # skip the 492-key benchmark

Outputs go to results/paper/ as CSV and JSON, and a summary is printed.

Experiments, in the order the paper reports them:
  E1  dataset counts and algorithm-win distribution
  E2  hidden-factor analysis (rho vs ECM by size of the smaller prime)
  E3  label separation (runner-up within 20% / 2x of the winner)
  E4  5-fold CV regression: RF 39 features / bit-length only / minus Fermat
  E5  within-tier Spearman correlation of RF predictions
  E6  algorithm selection vs size-only baselines, top-2, kappa, per-class recall
  E7  runtime portfolio analysis (oracle, top-1, parallel top-2, fixed choices)
  E8  FP(N) as a size proxy: correlation and residual AUC
  E9  oracle-augmented regression reference
  E10 cross-range generalization (train 32-96, test 97-160)
  E11 bootstrap confidence intervals
  E12 Fermat-vulnerable set: generation, benchmark, residual threshold
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, r2_score, mean_absolute_error,
                             roc_auc_score, recall_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.benchmarks import benchmark_semiprime_fast  # noqa: E402

SEED = 42
OUT = ROOT / "results" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

RF_PARAMS = dict(n_estimators=300, max_depth=10, max_features="sqrt",
                 bootstrap=True, random_state=SEED, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.06,
                  eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
ALGOS = ["pollard_rho", "pollard_pm1", "ecm", "fermat"]
CAPS = {"pollard_rho": 1.0, "pollard_pm1": 0.5, "ecm": 3.0, "fermat": 1.0}
REPORT = {}


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def load():
    df = pd.read_csv(ROOT / "data/processed/dataset_with_features_and_runtimes.csv")
    df = df.loc[:, ~df.columns.str.contains(r"\.\d+$")]  # drop duplicated columns
    hf = pd.read_csv(ROOT / "data/raw/hidden_factors.csv")
    df = df.merge(hf[["N", "p_hidden", "q_hidden", "p_bits", "q_bits"]], on="N", how="left")
    df["min_factor_bits"] = df[["p_bits", "q_bits"]].min(axis=1)
    df["residual"] = df["log_fermat_prox"] - 0.5 * df["log2_N"]
    return df


def feature_cols(df):
    drop = {"N", "structure", "best_algorithm", "min_runtime", "log_min_runtime",
            "p_hidden", "q_hidden", "p_bits", "q_bits", "min_factor_bits", "residual"}
    cols = [c for c in df.columns if c not in drop and not c.startswith(("runtime_", "success_", "timeout_"))]
    assert len(cols) == 39, f"expected 39 features, got {len(cols)}: {cols}"
    return cols


# ---------------------------------------------------------------- E1-E3
def e1_counts(df):
    section("E1 Dataset counts")
    n_total = len(df)
    fact = df[df.best_algorithm != "none"]
    wins = fact.best_algorithm.value_counts()
    tier = pd.cut(df.bit_length, [31, 50, 70, 90, 120, 160])
    timeout_rate = df.groupby(tier, observed=True).apply(lambda g: (g.best_algorithm == "none").mean())
    r = {"n_total": n_total, "n_factored": len(fact), "n_timeout": n_total - len(fact),
         "wins": wins.to_dict(), "win_share": (wins / len(fact)).round(4).to_dict(),
         "timeout_rate_by_tier": {str(k): round(v, 4) for k, v in timeout_rate.items()},
         "rho_share_32_50": round((fact[fact.bit_length <= 50].best_algorithm == "pollard_rho").mean(), 3),
         "ecm_share_gt90": round((fact[fact.bit_length > 90].best_algorithm == "ecm").mean(), 3)}
    print(json.dumps(r, indent=2)); REPORT["E1"] = r
    return fact.reset_index(drop=True)


def e2_hidden(fact):
    section("E2 Hidden-factor analysis: rho vs ECM by bits of the smaller prime")
    sub = fact[fact.best_algorithm.isin(["pollard_rho", "ecm"])]
    r = {"rho_share_minbits_le24": round((sub[sub.min_factor_bits <= 24].best_algorithm == "pollard_rho").mean(), 3),
         "rho_share_minbits_gt32": round((sub[sub.min_factor_bits > 32].best_algorithm == "pollard_rho").mean(), 3)}
    by = sub.groupby("min_factor_bits").best_algorithm.apply(lambda s: (s == "pollard_rho").mean()).round(3)
    by.to_csv(OUT / "E2_rho_share_by_min_factor_bits.csv")
    r["rho_share_by_min_factor_bits_25_35"] = by.loc[25:35].to_dict()
    print(json.dumps(r, indent=2)); REPORT["E2"] = r


def e3_separation(fact):
    section("E3 Label separation")
    rt = fact[[f"runtime_{a}" for a in ALGOS]].values.copy()
    ok = fact[[f"success_{a}" for a in ALGOS]].values.astype(bool)
    rt[~ok] = np.inf
    srt = np.sort(rt, axis=1)
    has2 = np.isfinite(srt[:, 1])
    ratio = srt[has2, 1] / srt[has2, 0]
    r = {"share_runner_up_within_20pct": round(float((ratio < 1.2).sum() / len(fact)), 4),
         "share_runner_up_within_2x": round(float((ratio < 2.0).sum() / len(fact)), 4),
         "share_min_runtime_below_1ms": round(float((fact.min_runtime < 1e-3).mean()), 4)}
    print(json.dumps(r, indent=2)); REPORT["E3"] = r


# ---------------------------------------------------------------- E4-E5
def cv_rf(X, y, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2s, maes = [], []
    for tr, te in kf.split(X):
        m = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr])
        p = m.predict(X[te]); r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
    out = {"model": label, "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
           "MAE_mean": np.mean(maes), "MAE_std": np.std(maes), "R2_folds": r2s}
    print(f"  {label:40s} R2 = {out['R2_mean']:.4f} ± {out['R2_std']:.4f}   MAE = {out['MAE_mean']:.3f}")
    return out


def e4_regression(fact, cols):
    section("E4 5-fold CV regression vs size-only baselines")
    y = fact.log_min_runtime.values
    fermat = ["fermat_proximity", "log_fermat_prox", "sqrt_fraction"]
    rows = [cv_rf(fact[cols].values, y, "RF 39 features"),
            cv_rf(fact[["bit_length"]].values, y, "RF bit_length only"),
            cv_rf(fact[[c for c in cols if c not in fermat]].values, y, "RF 39 minus Fermat features")]
    from scipy.stats import wilcoxon, ttest_rel
    a, b = rows[0]["R2_folds"], rows[1]["R2_folds"]
    stat = {"paired_t_p": float(ttest_rel(a, b).pvalue), "wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "mean_diff": float(np.mean(a) - np.mean(b))}
    print("  39 vs bit_length only:", stat)
    pd.DataFrame([{k: v for k, v in r.items() if k != "R2_folds"} for r in rows]).to_csv(OUT / "E4_regression_cv.csv", index=False)
    REPORT["E4"] = {"rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "R2_folds"} for r in rows], "test_39_vs_bitlength": stat}


def split(fact, cols):
    X = fact[cols].values; y_cls = fact.best_algorithm.values; y_reg = fact.log_min_runtime.values
    idx = np.arange(len(fact))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_cls)
    return X, y_cls, y_reg, tr, te


def e5_within_tier(fact, cols):
    section("E5 Within-tier Spearman of RF predictions (test set)")
    X, y_cls, y_reg, tr, te = split(fact, cols)
    m = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y_reg[tr]); p = m.predict(X[te])
    r = {"pooled_pearson": round(pearsonr(y_reg[te], p)[0], 3), "pooled_spearman": round(spearmanr(y_reg[te], p)[0], 3),
         "test_R2": round(r2_score(y_reg[te], p), 4), "test_MAE": round(mean_absolute_error(y_reg[te], p), 3)}
    tier = pd.cut(fact.bit_length.values[te], [31, 50, 70, 90, 120, 160])
    for t in tier.categories:
        mask = tier == t
        r[f"spearman_{t}"] = round(spearmanr(y_reg[te][mask], p[mask])[0], 3)
    print(json.dumps(r, indent=2)); REPORT["E5"] = r
    return p, te


# ---------------------------------------------------------------- E6-E7
def fit_xgb(Xtr, ytr, weighted=True):
    classes = np.unique(ytr); cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[c] for c in ytr])
    w = None
    if weighted:
        counts = np.bincount(yi); w = len(yi) / (len(classes) * counts[yi])
    m = XGBClassifier(**XGB_PARAMS).fit(Xtr, yi, sample_weight=w)
    return m, classes


def e6_selection(fact, cols):
    section("E6 Algorithm selection vs size-only baselines (test set)")
    X, y_cls, y_reg, tr, te = split(fact, cols)
    rows = []

    def add(name, pred):
        rows.append({"model": name, "accuracy": accuracy_score(y_cls[te], pred),
                     "balanced_accuracy": balanced_accuracy_score(y_cls[te], pred),
                     "macro_f1": f1_score(y_cls[te], pred, average="macro")})
        print(f"  {name:48s} acc={rows[-1]['accuracy']:.4f} bal={rows[-1]['balanced_accuracy']:.3f} mF1={rows[-1]['macro_f1']:.3f}")

    add("Always pollard_rho", np.full(len(te), "pollard_rho"))
    add("Always ecm", np.full(len(te), "ecm"))
    bl = fact.bit_length.values
    add("Rule: rho if bit_length<70 else ecm", np.where(bl[te] < 70, "pollard_rho", "ecm"))
    dt = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(bl[tr].reshape(-1, 1), y_cls[tr])
    add("Decision tree depth 3, bit_length only", dt.predict(bl[te].reshape(-1, 1)))
    X2 = fact[["log2_N", "residual"]].values
    m2, cl2 = fit_xgb(X2[tr], y_cls[tr]); add("XGBoost log2_N + scale-free residual", cl2[m2.predict(X2[te])])
    m, cl = fit_xgb(X[tr], y_cls[tr]); proba = m.predict_proba(X[te]); pred = cl[proba.argmax(1)]
    add("XGBoost 39 features (class-weighted)", pred)
    mu, clu = fit_xgb(X[tr], y_cls[tr], weighted=False); pred_u = clu[mu.predict(X[te])]
    add("XGBoost 39 features (unweighted)", pred_u)
    top2 = np.argsort(-proba, axis=1)[:, :2]
    yi = np.array([list(cl).index(c) for c in y_cls[te]])
    top2_acc = float(np.mean([yi[i] in top2[i] for i in range(len(te))]))
    rec = recall_score(y_cls[te], pred, labels=list(cl), average=None)
    r = {"rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for d in rows],
         "top2_accuracy": round(top2_acc, 4), "cohens_kappa": round(cohen_kappa_score(y_cls[te], pred), 3),
         "per_class_recall": {c: round(float(v), 3) for c, v in zip(cl, rec)}}
    print("  top-2 accuracy:", r["top2_accuracy"], " kappa:", r["cohens_kappa"], " recall:", r["per_class_recall"])
    pd.DataFrame(rows).to_csv(OUT / "E6_selection_baselines.csv", index=False); REPORT["E6"] = r
    return proba, cl, pred, pred_u, te


def e7_portfolio(fact, proba, cl, pred, pred_u, te):
    section("E7 Runtime portfolio analysis (test set)")
    sub = fact.iloc[te].reset_index(drop=True)
    rt = {a: sub[f"runtime_{a}"].values for a in ALGOS}
    ok = {a: sub[f"success_{a}"].values.astype(bool) for a in ALGOS}
    oracle = sub.min_runtime.values

    def cost_single(choice):
        c = np.where(ok[choice], rt[choice], CAPS[choice] + oracle) if isinstance(choice, str) else None
        return c

    def total_choice(choices):
        tot = 0.0
        for i, a in enumerate(choices):
            tot += rt[a][i] if ok[a][i] else CAPS[a] + oracle[i]
        return tot

    def total_parallel(c1, c2):
        tot = 0.0
        for i in range(len(c1)):
            t = [rt[a][i] if ok[a][i] else np.inf for a in (c1[i], c2[i])]
            best = min(t)
            tot += best if np.isfinite(best) else max(CAPS[c1[i]], CAPS[c2[i]]) + oracle[i]
        return tot

    order = np.argsort(-proba, axis=1)
    c1 = cl[order[:, 0]]; c2 = cl[order[:, 1]]
    T = {"oracle": float(oracle.sum()),
         "parallel_top2": total_parallel(c1, c2),
         "xgb_class_weighted_top1": total_choice(pred),
         "xgb_unweighted_top1": total_choice(pred_u),
         "always_ecm": total_choice(np.full(len(sub), "ecm")),
         "always_rho": total_choice(np.full(len(sub), "pollard_rho"))}

    def rec(strategy, base):
        return 100 * (T[base] - T[strategy]) / (T[base] - T["oracle"])

    r = {k: round(v, 1) for k, v in T.items()}
    r["recovered_vs_rho"] = {k: round(rec(k, "always_rho"), 1) for k in T if k not in ("oracle", "always_rho")}
    r["recovered_vs_ecm"] = {k: round(rec(k, "always_ecm"), 1) for k in T if k not in ("oracle", "always_ecm", "always_rho")}
    print(json.dumps(r, indent=2)); REPORT["E7"] = r


# ---------------------------------------------------------------- E8-E11
def e8_fp_proxy(fact):
    section("E8 FP(N) as a size proxy")
    y = (fact.best_algorithm == "fermat").astype(int)
    r = {"corr_logFP_log2N": round(pearsonr(fact.log_fermat_prox, fact.log2_N)[0], 4),
         "residual_median": round(float(fact.residual.median()), 2),
         "residual_auc_for_fermat_wins": round(roc_auc_score(y, -fact.residual), 3),
         "fermat_share_in_special": round((fact[fact.structure == "special"].best_algorithm == "fermat").mean(), 3)}
    print(json.dumps(r, indent=2)); REPORT["E8"] = r


def e9_oracle_augmented(fact, cols):
    section("E9 Oracle-augmented reference")
    X, y_cls, y_reg, tr, te = split(fact, cols)
    base = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y_reg[tr]).predict(X[te])
    oh = pd.get_dummies(fact.best_algorithm).values.astype(float)
    Xa = np.hstack([X, oh])
    aug = RandomForestRegressor(**RF_PARAMS).fit(Xa[tr], y_reg[tr]).predict(Xa[te])
    r = {"R2_test_39": round(r2_score(y_reg[te], base), 4), "R2_test_39_plus_label": round(r2_score(y_reg[te], aug), 4)}
    print(json.dumps(r, indent=2)); REPORT["E9"] = r


def e10_generalization(fact, cols):
    section("E10 Cross-range generalization (train 32-96 -> test 97-160)")
    small = fact.bit_length <= 96; large = ~small
    X = fact[cols].values; y_reg = fact.log_min_runtime.values; y_cls = fact.best_algorithm.values
    from xgboost import XGBRegressor
    rf = RandomForestRegressor(**RF_PARAMS).fit(X[small], y_reg[small]).predict(X[large])
    xg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06, random_state=SEED).fit(X[small], y_reg[small]).predict(X[large])
    m, cl = fit_xgb(X[small], y_cls[small]); pred = cl[m.predict(X[large])]
    r = {"n_train": int(small.sum()), "n_test": int(large.sum()),
         "R2_RF": round(r2_score(y_reg[large], rf), 3), "R2_XGB": round(r2_score(y_reg[large], xg), 3),
         "mean_logrt_train": round(float(y_reg[small].mean()), 2), "mean_logrt_test": round(float(y_reg[large].mean()), 2),
         "cls_accuracy": round(accuracy_score(y_cls[large], pred), 4), "cls_bal_acc": round(balanced_accuracy_score(y_cls[large], pred), 3),
         "always_ecm_accuracy": round(float((y_cls[large] == "ecm").mean()), 4),
         "classes_in_test": pd.Series(y_cls[large]).value_counts(normalize=True).round(3).to_dict()}
    print(json.dumps(r, indent=2)); REPORT["E10"] = r


def e11_bootstrap(fact, cols, n_boot=1000):
    section("E11 Bootstrap confidence intervals (test set)")
    X, y_cls, y_reg, tr, te = split(fact, cols)
    p = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y_reg[tr]).predict(X[te])
    m, cl = fit_xgb(X[tr], y_cls[tr]); pc = cl[m.predict(X[te])]
    rng = np.random.default_rng(SEED); r2s, accs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(te), len(te))
        r2s.append(r2_score(y_reg[te][i], p[i])); accs.append(accuracy_score(y_cls[te][i], pc[i]))
    r = {"R2_point": round(r2_score(y_reg[te], p), 3), "R2_ci": [round(np.percentile(r2s, 2.5), 3), round(np.percentile(r2s, 97.5), 3)],
         "acc_point": round(accuracy_score(y_cls[te], pc), 3), "acc_ci": [round(np.percentile(accs, 2.5), 3), round(np.percentile(accs, 97.5), 3)]}
    print(json.dumps(r, indent=2)); REPORT["E11"] = r


# ---------------------------------------------------------------- E12
def e12_fermat_set(fact, n_per_bits=41, bits=tuple(range(32, 161, 12)) + (160,)):
    section("E12 Fermat-vulnerable set (|q-p| < N^{1/4})")
    from sympy import randprime, nextprime
    import random
    bits = sorted(set(bits))[:12]
    rng = random.Random(SEED)
    rows = []
    t0 = time.perf_counter()
    for b in bits:
        for _ in range(n_per_bits):
            hb = b // 2
            p = randprime(2 ** (hb - 1), 2 ** hb)
            r = rng.randint(1, max(1, math.isqrt(p) // 2))
            q = nextprime(p + r)
            N = p * q
            bm = benchmark_semiprime_fast(N)
            fp = (math.isqrt(N) + (0 if math.isqrt(N) ** 2 == N else 1)) ** 2 - N
            d2 = ((q - p) // 2) ** 2
            rows.append({"N": N, "p": p, "q": q, "bits": N.bit_length(), "gap_over_N14": (q - p) / N ** 0.25,
                         "FP": fp, "d2": d2, "fp_equals_d2": fp == d2,
                         "residual": math.log2(fp + 1) - 0.5 * math.log2(N), **{k: v for k, v in bm.items() if k != "N"}})
        print(f"  {b:3d}-bit: done ({time.perf_counter() - t0:.0f}s)")
    vs = pd.DataFrame(rows); vs.to_csv(OUT / "E12_fermat_vulnerable_set.csv", index=False)
    thr = -3.5
    main_flag = fact.residual < thr
    fp_time = fact.loc[main_flag, "runtime_fermat"].sum()
    r = {"n": len(vs), "max_gap_over_N14": round(float(vs.gap_over_N14.max()), 3),
         "fp_equals_d2": f"{int(vs.fp_equals_d2.sum())}/{len(vs)}",
         "fermat_fastest": f"{int((vs.best_algorithm == 'fermat').sum())}/{len(vs)}",
         "fermat_runtime_median_us": round(float(vs.runtime_fermat.median() * 1e6), 1), "fermat_runtime_max_ms": round(float(vs.runtime_fermat.max() * 1e3), 3),
         "ecm_median_ms_le112": round(float(vs[vs.bits <= 112].runtime_ecm.median() * 1e3), 1),
         "ecm_timeout_share_ge128": round(float(vs[vs.bits >= 128].timeout_ecm.mean()), 3),
         "rho_success_share": round(float(vs.success_pollard_rho.mean()), 3), "ecm_success_share": round(float(vs.success_ecm.mean()), 3),
         "residual_min_med_max": [round(float(vs.residual.min()), 1), round(float(vs.residual.median()), 1), round(float(vs.residual.max()), 1)],
         "rule_recall_on_set": round(float((vs.residual < thr).mean()), 4),
         "rule_flagged_on_main": f"{int(main_flag.sum())}/{len(fact)} ({100 * main_flag.mean():.1f}%)",
         "fermat_time_on_main_false_positives_s": round(float(fp_time), 1)}
    print(json.dumps(r, indent=2)); REPORT["E12"] = r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-fermat", action="store_true"); ap.add_argument("--skip-bootstrap", action="store_true"); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    df = load(); cols = feature_cols(df)
    fact = e1_counts(df)
    run = lambda name: a.only is None or name in a.only
    if run("E2"): e2_hidden(fact)
    if run("E3"): e3_separation(fact)
    if run("E4"): e4_regression(fact, cols)
    if run("E5"): e5_within_tier(fact, cols)
    if run("E6") or run("E7"):
        proba, cl, pred, pred_u, te = e6_selection(fact, cols); e7_portfolio(fact, proba, cl, pred, pred_u, te)
    if run("E8"): e8_fp_proxy(fact)
    if run("E9"): e9_oracle_augmented(fact, cols)
    if run("E10"): e10_generalization(fact, cols)
    if run("E11") and not a.skip_bootstrap: e11_bootstrap(fact, cols)
    if run("E12") and not a.skip_fermat: e12_fermat_set(df)
    prev = json.load(open(OUT / "paper_numbers.json")) if (OUT / "paper_numbers.json").exists() else {}
    prev.update(REPORT)
    json.dump(prev, open(OUT / "paper_numbers.json", "w"), indent=2, default=float)
    print(f"\nAll results written to {OUT}")


if __name__ == "__main__":
    main()
