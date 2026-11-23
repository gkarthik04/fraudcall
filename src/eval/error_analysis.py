"""Error analysis for the main fused model F (seed 0): list call-level errors across the 14 test conditions and group by cause.
python -m src.eval.error_analysis      -> results/error_analysis.json, results/error_analysis.md
Causes (first match): recognition error (call WER > 0.40) | loud noise / hard channel (snr5, hard) | short call (< 30 s) |
normal call that sounds like a scam (false alarm on a label-0 call) | other (missed fraud with a clean transcript).
"""
import json, collections
import pandas as pd
from src.eval import metrics as M

MODEL, SEED = "F", 0
rows = {(r["cid"], r["cond"]): r for r in map(json.loads, open("data/manifest/manifest.jsonl", encoding="utf8")) if r["tts_set"] == 1}
conv = {c["cid"]: c for c in json.load(open("data/processed/conversations.json", encoding="utf8"))}
val = pd.read_csv(f"results/runs/{MODEL}/s{SEED}/preds_val.csv"); vf = M.final_scores(val[val.cond == "clean"])
thr = M.best_f1_threshold(vf.label.values, vf.score.values)
te = pd.read_csv(f"results/runs/{MODEL}/s{SEED}/preds_test.csv"); tf = M.final_scores(te[te.set == 1])
tf["pred"] = (tf.score >= thr).astype(int); err = tf[tf.pred != tf.label].copy()
out = []
for _, e in err.iterrows():
    r = rows[(e.cid, e.cond)]; c = conv[e.cid]
    if r["wer"] > 0.40: cause = "recognition error (call WER > 0.40)"
    elif e.cond in ("snr5", "hard"): cause = "loud noise / hard channel"
    elif r["duration"] < 30: cause = "short call (< 30 s)"
    elif e.label == 0: cause = "normal call that sounds like a scam (false alarm)"
    else: cause = "missed fraud with a good transcript"
    out.append({"cid": e.cid, "cond": e.cond, "label": int(e.label), "score": round(float(e.score), 3), "wer": round(r["wer"], 3), "duration": r["duration"],
                "case_type": c["case_type"], "domain": c["domain"], "scenario_id": c["scenario_id"], "pace": c["pace"], "cause": cause,
                "asr_head": r["asr_transcript"][:240], "clean_head": r["clean_transcript"][:240]})
n_all = len(tf)
summary = {"model": MODEL, "seed": SEED, "threshold": thr, "n_call_variants": n_all, "n_errors": len(out), "error_rate": len(out) / n_all,
           "by_cause": dict(collections.Counter(o["cause"] for o in out)), "by_case_type": dict(collections.Counter(o["case_type"] for o in out)),
           "by_label": dict(collections.Counter(o["label"] for o in out)), "by_cond": dict(collections.Counter(o["cond"] for o in out)),
           "by_domain": dict(collections.Counter(o["domain"] for o in out))}
json.dump({"summary": summary, "errors": out}, open("results/error_analysis.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
