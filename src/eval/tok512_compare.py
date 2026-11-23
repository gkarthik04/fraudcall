"""256 versus 512 text tokens (python -m src.eval.tok512_compare): T-asr (256) versus T-asr-tok512, seeds 0-2, set 1 test."""
import numpy as np
import pandas as pd
from src.eval.extra_analysis import per_seed, seeds_of, t_ci, md, TEST_CONDS_ALL

a, b = "T-asr-tok512", "T-asr"
seeds = [s for s in sorted(set(seeds_of(a)) & set(seeds_of(b))) if s <= 2]
A = {s: per_seed(a, s) for s in seeds}; B = {s: per_seed(b, s) for s in seeds}
deg = [c for c in TEST_CONDS_ALL if c != "clean"]
rows = []
for gname, conds in {"clean": ["clean"], "snr5": ["snr5"], "hard": ["hard"], "mean of 13 degraded": deg}.items():
    for mi, mn in ((0, "F1"), (1, "AUROC")):
        x = [np.mean([A[s][c][mi] for c in conds]) for s in seeds]; y = [np.mean([B[s][c][mi] for c in conds]) for s in seeds]
        lo, hi, p = t_ci(np.array(x) - np.array(y))
        rows.append({"condition": gname, "metric": mn, "n_seeds": len(seeds), "512 tokens": np.mean(x), "256 tokens": np.mean(y), "diff": np.mean(x) - np.mean(y), "ci95_lo": lo, "ci95_hi": hi, "p_paired_t": p})
T = pd.DataFrame(rows); T.to_csv("results/tok512_compare.csv", index=False)
open("results/tok512_compare.md", "w", encoding="utf8").write("# 512 vs 256 text tokens (T-asr, seeds %s)\n\n" % seeds + md(T)); print(md(T))
