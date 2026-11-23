"""Calibration and threshold analysis from saved predictions (python -m src.eval.calibration_threshold). Validation only for fitting.

1. Temperature scaling: fit one temperature T on validation call-level scores (the conditions the model was trained on), apply
   to test; report ECE before and after on clean / SNR 5 dB / hard test audio.
2. Threshold drift: the F1-optimal threshold chosen on clean validation (current practice) versus one chosen on pooled
   clean + randomly degraded validation calls (rand0, rand1). Reports test F1, recall and the share of normal calls flagged.
Writes results/calibration_temperature.csv, results/threshold_drift.csv, results/calibration_threshold.md.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from src.eval import metrics as M
from src.eval.extra_analysis import seeds_of, load, md

CONDS = ["clean", "snr5", "hard"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(y, s):
    z = logit(s); y = np.asarray(y)
    def nll(logT):
        q = 1 / (1 + np.exp(-z / np.exp(logT))); q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 4), method="bounded").x))


def apply_T(s, T):
    return 1 / (1 + np.exp(-logit(s) / T))


def main():
    cal, thr_rows = [], []
    for name in ("T-asr", "F"):
        for seed in seeds_of(name):
            val, test, tconds = load(name, seed)
            vf = M.final_scores(val[(val.set == 1) & val.cond.isin(tconds)])
            T = fit_temperature(vf.label.values, vf.score.values)
            tf = M.final_scores(test); tf = tf[tf.set == 1]
            for c in CONDS:
                g = tf[tf.cond == c]
                cal.append({"model": name, "seed": seed, "cond": c, "T": T, "ece_before": M.ece(g.label.values, g.score.values),
                            "ece_after": M.ece(g.label.values, apply_T(g.score.values, T))})
            # threshold drift
            v_clean = M.final_scores(val[(val.set == 1) & (val.cond == "clean")]); v_all = M.final_scores(val[(val.set == 1)])
            thr = {"clean-validation": M.best_f1_threshold(v_clean.label.values, v_clean.score.values),
                   "clean+degraded validation": M.best_f1_threshold(v_all.label.values, v_all.score.values)}
            for tn, th in thr.items():
                for c in CONDS:
                    g = tf[tf.cond == c]; m = M.basic_metrics(g.label.values, g.score.values, th)
                    thr_rows.append({"model": name, "seed": seed, "threshold from": tn, "thr": th, "cond": c, "f1": m["f1"], "recall": m["recall"], "normal_flagged": m["far_at_thr"]})
    C = pd.DataFrame(cal); C.to_csv("results/calibration_temperature.csv", index=False)
    D = pd.DataFrame(thr_rows); D.to_csv("results/threshold_drift.csv", index=False)
    Cs = C.groupby(["model", "cond"]).agg(T=("T", "mean"), ece_before=("ece_before", "mean"), ece_after=("ece_after", "mean"), n_seeds=("seed", "count")).reset_index()
    Ds = D.groupby(["model", "threshold from", "cond"]).agg(thr=("thr", "mean"), f1=("f1", "mean"), recall=("recall", "mean"), normal_flagged=("normal_flagged", "mean"), n_seeds=("seed", "count")).reset_index()
    txt = ["# Calibration and threshold analysis (set 1 test; fitted on validation only; mean over seeds)\n", "## Temperature scaling (ECE, lower is better)\n", md(Cs),
           "\n## Threshold chosen on clean validation versus clean + degraded validation\n", md(Ds)]
    open("results/calibration_threshold.md", "w", encoding="utf8").write("\n".join(txt)); print("\n".join(txt))


if __name__ == "__main__":
    main()
