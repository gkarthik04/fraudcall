"""Evaluation metrics. Fraud = positive class.

Call-level score  = the model's score at the LAST step of the call (it has then heard the whole call, capped at 60 s).
Operating threshold = chosen on VALIDATION only (max call-level F1 over the validation conditions the model was trained on).
Recall@5%FAR   = best recall with false-alarm rate <= 5 % on the evaluated condition's own ROC curve (a rank metric).
Time-to-detection threshold = the lowest threshold at which at most 5 % of VALIDATION normal calls ever fire at any step.
Confidence intervals: cluster bootstrap over session_id (calls of one session are not independent).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support

STEP_GRID = [10, 20, 30, 40, 50, 60]


def load_preds(path):
    return pd.read_csv(path)


def final_scores(df):
    """One row per (set, cond, cid): score at the last step, max score, label."""
    df = df.sort_values(["set", "cond", "cid", "step"])
    last = df.groupby(["set", "cond", "cid"], as_index=False).tail(1)[["set", "cond", "cid", "label", "score", "t"]]
    mx = df.groupby(["set", "cond", "cid"], as_index=False)["score"].max().rename(columns={"score": "max_score"})
    return last.merge(mx, on=["set", "cond", "cid"])


def best_f1_threshold(y, s):
    cands = np.unique(np.concatenate([s, [0.5]]))
    best, bt = -1, 0.5
    for t in cands:
        p = s >= t
        tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = ((~p) & (y == 1)).sum()
        f = 2 * tp / max(1, 2 * tp + fp + fn)
        if f > best + 1e-12:
            best, bt = f, t
    return float(bt)


def recall_at_far(y, s, far=0.05):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= far].max())


def basic_metrics(y, s, thr):
    y = np.asarray(y); s = np.asarray(s)
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    ok = len(set(y)) == 2
    return {"precision": pr, "recall": rc, "f1": f1,
            "auroc": roc_auc_score(y, s) if ok else float("nan"),
            "pr_auc": average_precision_score(y, s) if ok else float("nan"),
            "recall_at_5far": recall_at_far(y, s), "far_at_thr": float(((p == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))}


def ece(y, s, bins=10):
    y = np.asarray(y); s = np.asarray(s); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if m.any():
            e += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(e)


def reliability(y, s, bins=10):
    y = np.asarray(y); s = np.asarray(s); edges = np.linspace(0, 1, bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if m.any():
            out.append((float(s[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def far_threshold(val_final, far=0.05):
    """Smallest threshold such that <= far of normal validation calls have max-over-steps score >= it."""
    neg = np.sort(val_final.loc[val_final.label == 0, "max_score"].values)
    k = int(np.floor(far * len(neg)))                 # allowed number of false alarms
    return float(neg[-(k + 1)] + 1e-9) if len(neg) else 0.5


def detection_times(df_cond, thr):
    """For each call, the first step time where score >= thr (np.inf if never). Returns DataFrame cid,label,ttd."""
    d = df_cond.sort_values(["cid", "step"]); rows = []
    for cid, g in d.groupby("cid"):
        hit = g[g.score >= thr]
        rows.append({"cid": cid, "label": int(g.label.iloc[0]), "ttd": float(hit.t.iloc[0]) if len(hit) else np.inf})
    return pd.DataFrame(rows)


def flagged_curve(ttd_df, grid=STEP_GRID):
    pos = ttd_df[ttd_df.label == 1]
    return [float((pos.ttd <= t).mean()) if len(pos) else float("nan") for t in grid]


def cluster_bootstrap(df_final, session_of, fn, n=500, seed=0):
    """Bootstrap by session_id. fn(DataFrame) -> float. Returns (lo, hi) 95% percentile CI."""
    rng = np.random.default_rng(seed)
    sess = df_final.cid.map(session_of).values; us = np.unique(sess)
    groups = {s: np.where(sess == s)[0] for s in us}; vals = []
    for _ in range(n):
        pick = rng.choice(us, size=len(us), replace=True)
        idx = np.concatenate([groups[s] for s in pick])
        v = fn(df_final.iloc[idx])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
