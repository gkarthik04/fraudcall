"""Extra analyses (python -m src.eval.extra_analysis). Reads saved predictions only; thresholds come from validation.

1. Paired tests of H2: F (fusion) minus T-asr (text only) over every seed both models share (seeds 0-4 when trained),
   per condition, for F1 and AUROC: mean difference, paired-t 95 % CI over seeds, and a session-cluster bootstrap CI of the
   AUROC difference of the seed-averaged call scores.
2. "k consecutive steps" alert rule (H4 follow-up): alert when the score stays >= a threshold for k steps in a row; the
   threshold is chosen on validation normal calls so that at most 5 % of them ever alert. k = 1 is the rule used so far.
Writes results/paired_tests.csv, results/alert_rule.csv, results/extra_analysis.md.
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
from src.eval import metrics as M

TEST_CONDS_ALL = ["clean", "bw8k", "mulaw", "alaw", "amrnb", "gsm", "opus12", "opus8", "loss5", "loss10", "snr20", "snr10", "snr5", "hard"]


def seeds_of(name):
    return sorted(int(p.split("\\s")[-1].split("/s")[-1].split(os.sep)[0].replace("s", "")) for p in
                  [os.path.dirname(x) for x in glob.glob(f"results/runs/{name}/s*/summary.json")])


def load(name, seed):
    d = f"results/runs/{name}/s{seed}"
    return pd.read_csv(f"{d}/preds_val.csv"), pd.read_csv(f"{d}/preds_test.csv"), json.load(open(f"{d}/summary.json"))["exp"]["train_conds"]


def per_seed(name, seed):
    val, test, tconds = load(name, seed)
    vf = M.final_scores(val[val.cond.isin(tconds)]); thr = M.best_f1_threshold(vf.label.values, vf.score.values)
    tf = M.final_scores(test); tf = tf[tf.set == 1]
    out = {}
    for c, g in tf.groupby("cond"):
        m = M.basic_metrics(g.label.values, g.score.values, thr); out[c] = (m["f1"], m["auroc"])
    return out


def t_ci(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    p = stats.ttest_1samp(d, 0.0).pvalue if d.std(ddof=1) > 0 else float("nan")
    return float(d.mean() - h), float(d.mean() + h), float(p)


def paired(a="F", b="T-asr"):
    seeds = sorted(set(seeds_of(a)) & set(seeds_of(b)))
    A = {s: per_seed(a, s) for s in seeds}; B = {s: per_seed(b, s) for s in seeds}
    sess = {c["cid"]: c["session_id"] for c in json.load(open("data/processed/conversations.json", encoding="utf8"))}
    rows = []
    degraded = [c for c in TEST_CONDS_ALL if c != "clean"]
    groups = {c: [c] for c in ["clean", "snr10", "snr5", "loss10", "opus8", "hard"]}; groups["mean of 13 degraded"] = degraded
    for gname, conds in groups.items():
        for mi, mname in ((0, "F1"), (1, "AUROC")):
            d = [np.mean([A[s][c][mi] - B[s][c][mi] for c in conds]) for s in seeds]
            lo, hi, p = t_ci(d)
            row = {"condition": gname, "metric": mname, "n_seeds": len(seeds), f"{a}": np.mean([np.mean([A[s][c][mi] for c in conds]) for s in seeds]),
                   f"{b}": np.mean([np.mean([B[s][c][mi] for c in conds]) for s in seeds]), "diff": float(np.mean(d)), "ci95_lo": lo, "ci95_hi": hi, "p_paired_t": p}
            rows.append(row)
    # call-level cluster bootstrap of the AUROC difference on seed-averaged scores (clean and hard)
    boot = []
    for c in ["clean", "snr5", "hard"]:
        fa, fb = [], []
        for s in seeds:
            for nm, store in ((a, fa), (b, fb)):
                _, test, _ = load(nm, s); tf = M.final_scores(test); tf = tf[(tf.set == 1) & (tf.cond == c)].set_index("cid")
                store.append(tf)
        sa = pd.concat([x.score for x in fa], axis=1).mean(axis=1); sb = pd.concat([x.score for x in fb], axis=1).mean(axis=1)
        lab = fa[0].label.reindex(sa.index)
        df = pd.DataFrame({"cid": sa.index, "label": lab.values, "sa": sa.values, "sb": sb.reindex(sa.index).values})
        fn = lambda g: (M.basic_metrics(g.label.values, g.sa.values, .5)["auroc"] - M.basic_metrics(g.label.values, g.sb.values, .5)["auroc"])
        lo, hi = M.cluster_bootstrap(df, sess, fn, n=500)
        boot.append({"condition": c, "metric": "AUROC (seed-averaged scores)", "diff": fn(df), "boot_ci95_lo": lo, "boot_ci95_hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(boot), seeds


def k_rule_threshold(val, k, far=0.05):
    """Smallest threshold so that <= far of validation normal calls have k consecutive steps >= it."""
    def run_max(g):                                    # best 'min over k consecutive steps' of a call
        s = g.sort_values("step").score.values
        return max((s[i:i + k].min() for i in range(max(1, len(s) - k + 1))), default=0.0)
    neg = np.sort(np.array([run_max(g) for _, g in val[val.label == 0].groupby("cid")]))
    kk = int(np.floor(far * len(neg)))
    return float(neg[-(kk + 1)] + 1e-9)


def k_alert_times(df, thr, k):
    rows = []
    for cid, g in df.sort_values(["cid", "step"]).groupby("cid"):
        s = g.score.values; t = g.t.values; hit = np.inf
        for i in range(len(s) - k + 1):
            if (s[i:i + k] >= thr).all():
                hit = float(t[i + k - 1]); break
        rows.append({"cid": cid, "label": int(g.label.iloc[0]), "ttd": hit})
    return pd.DataFrame(rows)


def alert_rule(models=("T-asr", "F"), conds=("clean", "snr5", "hard")):
    rows = []
    for name in models:
        for seed in seeds_of(name):
            val, test, tconds = load(name, seed); val = val[(val.cond.isin(tconds)) & (val.set == 1)]
            for k in (1, 2, 3):
                thr = k_rule_threshold(val, k)
                for c in conds:
                    g = test[(test.set == 1) & (test.cond == c)]
                    d = k_alert_times(g, thr, k); pos = d[d.label == 1]; neg = d[d.label == 0]
                    rows.append({"model": name, "seed": seed, "k": k, "cond": c, "thr": thr, "far_realized": float(np.isfinite(neg.ttd).mean()),
                                 "flagged_final": float(np.isfinite(pos.ttd).mean()),
                                 **{f"flag@{t}": float((pos.ttd <= t).mean()) for t in (20, 30, 40, 60)},
                                 "mean_ttd_missed_as_60": float(pos.ttd.replace(np.inf, 60.0).mean())})
    return pd.DataFrame(rows)


def md(df, floatfmt=3):
    df = df.copy(); cols = list(df.columns)
    def f(v):
        return f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    pairs = [("F", "T-asr", "H2: F (fusion) minus T-asr (text only)"), ("T-asr-aug", "T-asr", "H3: T-asr-aug minus T-asr (augmentation, text only)"),
             ("F-aug", "F", "H3: F-aug minus F (augmentation, fusion)")]
    txt = []; allp = []
    for a_, b_, title in pairs:
        if len(set(seeds_of(a_)) & set(seeds_of(b_))) < 3:
            continue
        P, B, seeds = paired(a_, b_); P.insert(0, "pair", f"{a_} - {b_}"); allp.append(P)
        txt += [f"## {title}, set 1 test, seeds {seeds}\n", md(P.drop(columns="pair")),
                "\nPaired-t CI is over seeds (n = shared seeds, so wide). Session-cluster bootstrap of the seed-averaged AUROC difference:\n", md(B, 4), ""]
    pd.concat(allp).to_csv("results/paired_tests.csv", index=False)
    R = alert_rule(); R.to_csv("results/alert_rule.csv", index=False)
    S = R.groupby(["model", "k", "cond"]).agg(far=("far_realized", "mean"), flagged_final=("flagged_final", "mean"), f30=("flag@30", "mean"), f40=("flag@40", "mean"),
                                             f60=("flag@60", "mean"), mean_ttd=("mean_ttd_missed_as_60", "mean"), thr=("thr", "mean")).reset_index()
    S.to_csv("results/alert_rule_summary.csv", index=False)
    txt = ["# Extra analyses\n"] + txt + ["## Alert rule: k consecutive steps above a validation threshold (<= 5 % of normal validation calls alert), mean over all seeds\n",
           "flagged@t = share of fraud calls alerted by t seconds; far = share of normal test calls alerted; mean_ttd counts never-alerted fraud calls as 60 s.\n", md(S)]
    open("results/extra_analysis.md", "w", encoding="utf8").write("\n".join(txt))
    print("\n".join(txt[:30]))


if __name__ == "__main__":
    main()
