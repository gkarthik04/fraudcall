"""Compute all metric tables from saved predictions.  python -m src.eval.run_eval

Reads  results/runs/{model}/s{seed}/preds_{val,test}.csv  (+ summary.json for the training conditions)
Writes results/metrics_long.csv       per model, seed, TTS set, condition: F1, AUROC, ... , ECE
       results/summary_by_cond.csv    mean / std over seeds
       results/ttd.csv                time-to-detection at 5 % false alarms
       results/bootstrap_ci.csv       session-cluster bootstrap 95 % CIs of seed-averaged F1 / AUROC
       results/reliability.json       reliability-diagram bins for the main models
       results/wer_by_cond.csv        word error rate per channel condition
"""
import glob, json, os
import numpy as np
import pandas as pd
from src.common import load_config, write_json
from src.eval import metrics as M

KEYS = ["set", "cond", "cid", "step"]
LATE = {"F-late": ("T-asr", "A"), "F-late-aug": ("T-asr-aug", "A-aug")}


def runs():
    out = {}
    for p in glob.glob("results/runs/*/s*/summary.json"):
        s = json.load(open(p)); out.setdefault(s["name"], {})[s["seed"]] = s
    return out


def get_preds(name, seed, R):
    if name in LATE:
        a, b = LATE[name]
        res = []
        for part in ("val", "test"):
            x = pd.read_csv(f"results/runs/{a}/s{seed}/preds_{part}.csv"); y = pd.read_csv(f"results/runs/{b}/s{seed}/preds_{part}.csv")
            m = x.merge(y[KEYS + ["score"]], on=KEYS, suffixes=("", "_b")); m["score"] = (m["score"] + m["score_b"]) / 2
            res.append(m.drop(columns="score_b"))
        return res[0], res[1], R[a][seed]["exp"]["train_conds"]
    return (pd.read_csv(f"results/runs/{name}/s{seed}/preds_val.csv"), pd.read_csv(f"results/runs/{name}/s{seed}/preds_test.csv"),
            R[name][seed]["exp"]["train_conds"])


def main():
    cfg = load_config(); R = runs()
    session_of = {c["cid"]: c["session_id"] for c in json.load(open("data/processed/conversations.json", encoding="utf8"))}
    names = sorted(set(R) | {k for k, (a, b) in LATE.items() if a in R and b in R})
    long, ttd, rel = [], [], {}
    for name in names:
        seeds = sorted(R[LATE[name][0]] if name in LATE else R[name])
        for seed in seeds:
            val, test, tconds = get_preds(name, seed, R)
            vf = M.final_scores(val[val.cond.isin(tconds)]); thr = M.best_f1_threshold(vf.label.values, vf.score.values)
            thr5 = M.far_threshold(vf, 0.05)
            tf = M.final_scores(test)
            for (s, c), g in tf.groupby(["set", "cond"]):
                m = M.basic_metrics(g.label.values, g.score.values, thr)
                m.update(model=name, seed=seed, set=s, cond=c, thr=thr, ece=M.ece(g.label.values, g.score.values), n=len(g))
                long.append(m)
                if name in ("T-asr", "A", "F", "F-aug", "T-asr-aug") and c in ("clean", "hard"):
                    rel[f"{name}|s{seed}|set{s}|{c}"] = M.reliability(g.label.values, g.score.values)
            for (s, c), g in test.groupby(["set", "cond"]):
                d = M.detection_times(g, thr5); neg = d[d.label == 0]; pos = d[d.label == 1]
                fl = M.flagged_curve(d)
                ttd.append({"model": name, "seed": seed, "set": s, "cond": c, "thr5": thr5,
                            "far_realized": float(np.isfinite(neg.ttd).mean()) if len(neg) else np.nan,
                            "median_ttd_flagged": float(pos.ttd[np.isfinite(pos.ttd)].median()) if np.isfinite(pos.ttd).any() else np.nan,
                            "mean_ttd_missed_as_60": float(pos.ttd.replace(np.inf, 60.0).mean()) if len(pos) else np.nan,
                            **{f"flag@{t}": v for t, v in zip(M.STEP_GRID, fl)}})
    L = pd.DataFrame(long); T = pd.DataFrame(ttd)
    L.to_csv("results/metrics_long.csv", index=False); T.to_csv("results/ttd.csv", index=False)
    cols = ["f1", "auroc", "pr_auc", "recall_at_5far", "precision", "recall", "far_at_thr", "ece"]
    S = L.groupby(["model", "set", "cond"])[cols].agg(["mean", "std"]); S.columns = ["_".join(c) for c in S.columns]
    S.reset_index().to_csv("results/summary_by_cond.csv", index=False)
    # bootstrap CI on seed-averaged scores (calls of a session resampled together)
    ci = []
    for name in names:
        seeds = sorted(R[LATE[name][0]] if name in LATE else R[name]); parts = []
        for seed in seeds:
            val, test, tconds = get_preds(name, seed, R)
            vf = M.final_scores(val[val.cond.isin(tconds)]); thr = M.best_f1_threshold(vf.label.values, vf.score.values)
            tf = M.final_scores(test); tf["seed"] = seed; tf["thr"] = thr; parts.append(tf)
        allp = pd.concat(parts)
        for (s, c), g in allp.groupby(["set", "cond"]):
            avg = g.groupby("cid").agg(label=("label", "first"), score=("score", "mean")).reset_index()
            thr = float(g.thr.mean())
            f1 = lambda d: M.basic_metrics(d.label.values, d.score.values, thr)["f1"]
            au = lambda d: M.basic_metrics(d.label.values, d.score.values, thr)["auroc"]
            lo, hi = M.cluster_bootstrap(avg, session_of, f1, n=300); alo, ahi = M.cluster_bootstrap(avg, session_of, au, n=300)
            ci.append({"model": name, "set": s, "cond": c, "f1_ens": f1(avg), "f1_lo": lo, "f1_hi": hi, "auroc_ens": au(avg), "auroc_lo": alo, "auroc_hi": ahi, "n_calls": len(avg)})
    pd.DataFrame(ci).to_csv("results/bootstrap_ci.csv", index=False)
    write_json(rel, "results/reliability.json")
    # WER per condition
    rows = [json.loads(l) for l in open(cfg["paths"]["manifest"] + "/manifest.jsonl", encoding="utf8")]
    W = pd.DataFrame([{"set": r["tts_set"], "cond": r["cond"], "split": r["split"], "wer": r["wer"]} for r in rows])
    W = W[~W.cond.str.startswith("rand")]
    W.groupby(["set", "cond"]).wer.agg(["mean", "median", "count"]).reset_index().to_csv("results/wer_by_cond.csv", index=False)
    print("models:", names); print(S.reset_index().query("set == 1 and cond in ['clean','hard']").round(3).to_string())


if __name__ == "__main__":
    main()
