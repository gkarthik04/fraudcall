"""Generate every figure and markdown table from results/*.csv.  python -m src.eval.figures"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"T-clean": "#7f7f7f", "T-asr": "#1f77b4", "T-asr-aug": "#6baed6", "A": "#d95f02", "A-aug": "#fdae6b",
       "F": "#1b9e77", "F-aug": "#7570b3", "F-late": "#e7298a", "F-late-aug": "#e6ab02", "T-rob": "#8c564b"}
GRID = ["clean", "bw8k", "mulaw", "alaw", "amrnb", "gsm", "opus12", "opus8", "loss5", "loss10", "snr20", "snr10", "snr5", "hard"]
os.makedirs("figures", exist_ok=True)


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def ms(S, model, st, cond, col):
    r = S[(S.model == model) & (S.set == st) & (S.cond == cond)]
    if r.empty:
        return np.nan, np.nan
    return float(r[col + "_mean"].iloc[0]), float(r[col + "_std"].iloc[0])


def main():
    S = pd.read_csv("results/summary_by_cond.csv"); T = pd.read_csv("results/ttd.csv")
    W = pd.read_csv("results/wer_by_cond.csv"); CI = pd.read_csv("results/bootstrap_ci.csv")
    models = set(S.model); tables = []
    wer1 = W[W.set == 1].set_index("cond")["mean"]
    order = sorted([c for c in GRID if c in wer1.index], key=lambda c: (c != "clean", wer1[c]))
    # ---- WER by condition
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.bar(order, [wer1[c] for c in order], color="#4c78a8")
    ax.set_ylabel("word error rate"); ax.set_title("Speech-recognition error by phone-channel condition (set 1, test calls)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig("figures/wer_by_condition.png", dpi=150); plt.close()
    tables.append("## WER per condition (test split)\n\n" + md(W[W.split.isna()] if "split" in W else W))
    # ---- H1: F1 vs severity
    h1 = [m for m in ("T-clean", "T-asr", "A", "F") if m in models]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for m in h1:
        y = [ms(S, m, 1, c, "f1")[0] for c in order]; e = [ms(S, m, 1, c, "f1")[1] for c in order]
        ax.errorbar(range(len(order)), y, yerr=e, marker="o", label=m, color=PAL[m], capsize=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{c}\n({wer1[c]:.2f})" for c in order], fontsize=7)
    ax.set_ylabel("call-level F1 (fraud = positive)"); ax.set_xlabel("condition (WER of recognised text), ordered by severity")
    ax.set_title("H1/H2: clean-trained models across the phone-channel grid (mean ± sd, 3 seeds)"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("figures/h1_f1_vs_severity.png", dpi=150); plt.close()
    rows = []
    for c in order:
        r = {"condition": c, "WER": wer1[c]}
        for m in [x for x in ("T-clean", "T-asr", "A", "F") if x in models]:
            f, fs = ms(S, m, 1, c, "f1"); a, _ = ms(S, m, 1, c, "auroc"); r[f"{m} F1"] = f"{f:.3f}±{fs:.3f}"; r[f"{m} AUROC"] = f"{a:.3f}"
        rows.append(r)
    tables.append("## H1 full grid (clean-trained models, set 1 test, mean±sd of F1 over 3 seeds)\n\n" + md(pd.DataFrame(rows)))
    # ---- H2 table (F1, AUROC, recall@5%FAR) for key conditions
    key = ["clean", "gsm", "snr10", "snr5", "hard"]; rows = []
    for m in [x for x in ["T-clean", "T-asr", "A", "F", "F-late", "T-asr-aug", "A-aug", "F-aug", "F-late-aug", "T-rob"] if x in models]:
        r = {"model": m}
        for c in key:
            for col, nm in (("f1", "F1"), ("auroc", "AUROC"), ("recall_at_5far", "R@5%FAR")):
                a, b = ms(S, m, 1, c, col); r[f"{c} {nm}"] = f"{a:.3f}±{b:.3f}" if not np.isnan(a) else "-"
        rows.append(r)
    tables.append("## H2/H3 main table: call-level metrics, mean±sd over 3 seeds (set 1 test)\n\n" + md(pd.DataFrame(rows)))
    # grid-mean over degraded conditions
    deg = [c for c in order if c != "clean"]; rows = []
    for m in sorted(models):
        v = [ms(S, m, 1, c, "f1")[0] for c in deg]; au = [ms(S, m, 1, c, "auroc")[0] for c in deg]
        rows.append({"model": m, "mean F1 over 13 degraded conds": np.nanmean(v), "mean AUROC over 13 degraded conds": np.nanmean(au),
                     "clean F1": ms(S, m, 1, "clean", "f1")[0], "clean AUROC": ms(S, m, 1, "clean", "auroc")[0]})
    G = pd.DataFrame(rows); G.to_csv("results/grid_mean.csv", index=False)
    tables.append("## Mean over the degraded grid\n\n" + md(G))
    # ---- H3: hard conditions bar chart
    hard = [c for c in ("opus8", "snr5", "loss10", "hard") if c in order]; pair = [m for m in ("F", "F-aug", "T-asr", "T-asr-aug") if m in models]
    fig, ax = plt.subplots(figsize=(8, 3.8)); w = 0.8 / max(1, len(pair))
    for i, m in enumerate(pair):
        y = [ms(S, m, 1, c, "f1")[0] for c in hard]; e = [ms(S, m, 1, c, "f1")[1] for c in hard]
        ax.bar(np.arange(len(hard)) + i * w, y, w, yerr=e, label=m, color=PAL[m], capsize=2)
    ax.set_xticks(np.arange(len(hard)) + w * (len(pair) - 1) / 2); ax.set_xticklabels(hard); ax.set_ylabel("call-level F1")
    ax.set_title("H3: channel augmentation on the hardest conditions"); ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig("figures/h3_augmentation_hard.png", dpi=150); plt.close()
    # ---- H4: flagged share vs seconds
    secs = [10, 20, 30, 40, 50, 60]
    for cond in ("clean", "hard"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in [x for x in ["T-asr", "A", "F", "T-asr-aug", "A-aug", "F-aug"] if x in models]:
            g = T[(T.model == m) & (T.set == 1) & (T.cond == cond)]
            if g.empty:
                continue
            y = [g[f"flag@{s}"].mean() for s in secs]
            ax.plot(secs, y, marker="o", label=m, color=PAL[m], ls="--" if m.endswith("aug") else "-")
        ax.set_xlabel("seconds into the call"); ax.set_ylabel("share of fraud calls flagged")
        ax.set_title(f"H4: early detection at 5% false alarms ({cond})"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"figures/h4_time_to_detection_{cond}.png", dpi=150); plt.close()
    rows = []
    for m in [x for x in ["T-asr", "A", "F", "T-asr-aug", "A-aug", "F-aug", "F-late-aug"] if x in models]:
        for cond in ("clean", "hard"):
            g = T[(T.model == m) & (T.set == 1) & (T.cond == cond)]
            if g.empty:
                continue
            rows.append({"model": m, "cond": cond, "val-chosen thr": g.thr5.mean(), "realised false-alarm rate (test)": f"{g.far_realized.mean():.3f}±{g.far_realized.std():.3f}",
                         "flagged by 20 s": g["flag@20"].mean(), "flagged by 40 s": g["flag@40"].mean(), "flagged by 60 s": g["flag@60"].mean(),
                         "median time flagged calls (s)": g.median_ttd_flagged.mean(), "mean TTD, missed=60 s": g.mean_ttd_missed_as_60.mean()})
    tables.append("## H4: time-to-detection at 5 % false-alarm threshold (threshold chosen on validation normal calls)\n\n" + md(pd.DataFrame(rows)))
    # ---- H5: set1 vs set2
    conds2 = ["clean", "gsm", "snr10", "hard"]; rows = []
    show = [m for m in ("T-asr", "A", "F", "T-asr-aug", "A-aug", "F-aug") if m in models]
    fig, axes = plt.subplots(1, len(conds2), figsize=(12, 3.5), sharey=True)
    for ax, c in zip(axes, conds2):
        x = np.arange(len(show))
        s1 = [ms(S, m, 1, c, "f1")[0] for m in show]; s2 = [ms(S, m, 2, c, "f1")[0] for m in show]
        ax.bar(x - 0.2, s1, 0.4, label="Set 1 (Piper)", color="#4c78a8"); ax.bar(x + 0.2, s2, 0.4, label="Set 2 (MMS, unseen)", color="#f58518")
        ax.set_xticks(x); ax.set_xticklabels(show, rotation=60, fontsize=7); ax.set_title(c)
        for m, a, b in zip(show, s1, s2):
            rows.append({"model": m, "cond": c, "F1 set1": a, "F1 set2": b, "drop": a - b})
    axes[0].set_ylabel("call-level F1"); axes[0].legend(fontsize=7); plt.suptitle("H5: same model, different TTS system", y=1.02)
    plt.tight_layout(); plt.savefig("figures/h5_unseen_tts.png", dpi=150, bbox_inches="tight"); plt.close()
    tables.append("## H5: F1 on TTS Set 1 vs unseen Set 2 (same test calls)\n\n" + md(pd.DataFrame(rows)))
    # ---- ablations
    abl = [m for m in ("F-aug", "T-asr-aug", "A-aug", "F-late-aug", "F-aug-win10", "F-aug-win60", "F-aug-tok128", "F", "F-late", "T-asr", "A") if m in models]
    rows = []
    for m in abl:
        d = [c for c in order if c != "clean"]
        rows.append({"model": m, "clean F1": ms(S, m, 1, "clean", "f1")[0], "mean F1 degraded": np.nanmean([ms(S, m, 1, c, "f1")[0] for c in d]),
                     "mean AUROC degraded": np.nanmean([ms(S, m, 1, c, "auroc")[0] for c in d]), "hard F1": ms(S, m, 1, "hard", "f1")[0], "hard AUROC": ms(S, m, 1, "hard", "auroc")[0]})
    tables.append("## Ablations (3 seeds). Remove audio = T-asr(-aug); remove text = A(-aug); late vs early fusion = F-late vs F; window 10/30/60 s = F-aug-win10 / F-aug / F-aug-win60; text 128/256 tokens = F-aug-tok128 / F-aug; augmentation on/off = *-aug vs plain\n\n" + md(pd.DataFrame(rows)))
    # ---- calibration
    if os.path.exists("results/reliability.json"):
        rel = json.load(open("results/reliability.json")); fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, c in zip(axes, ("clean", "hard")):
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            for m in ("T-asr", "A", "F", "F-aug"):
                k = f"{m}|s0|set1|{c}"
                if k in rel and rel[k]:
                    b = np.array(rel[k]); ax.plot(b[:, 0], b[:, 1], marker="o", label=m, color=PAL[m])
            ax.set_title(f"reliability ({c}, seed 0)"); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraud rate"); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig("figures/calibration_reliability.png", dpi=150); plt.close()
        rows = [{"model": m, "ECE clean": ms(S, m, 1, "clean", "ece")[0], "ECE hard": ms(S, m, 1, "hard", "ece")[0]} for m in sorted(models)]
        tables.append("## Expected calibration error (10 bins, call level)\n\n" + md(pd.DataFrame(rows)))
    # ---- bootstrap CI table
    c = CI[(CI.set == 1) & (CI.cond.isin(["clean", "snr5", "hard"]))].copy()
    c["F1 (95% CI)"] = c.apply(lambda r: f"{r.f1_ens:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]", axis=1)
    c["AUROC (95% CI)"] = c.apply(lambda r: f"{r.auroc_ens:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]", axis=1)
    tables.append("## Bootstrap CIs (session-cluster, seed-averaged scores)\n\n" + md(c[["model", "cond", "n_calls", "F1 (95% CI)", "AUROC (95% CI)"]]))
    open("results/tables.md", "w", encoding="utf8").write("\n\n".join(tables))
    print("figures and tables written")


if __name__ == "__main__":
    main()
