"""Prepare the demo: model checkpoints + thresholds (from validation) and 6 held-out test calls.

python -m scripts.make_demo   (needs results/metrics_long.csv and results/ttd.csv from src.eval.run_eval)
Samples: 3 fraud + 3 normal test-split calls; one is a "hard case" (a fraud call the fused model misses at 60 s
or, if none, the lowest-margin call). Chosen with a fixed seed from calls the fused model got right, except the hard case.
"""
import json, os, shutil
import numpy as np, pandas as pd
from src.common import write_json
from src.eval import metrics as M

SEED = 0
MODELS = {"text": ("T-asr-aug", "text"), "audio": ("A-aug", "audio"), "fused": ("F-aug", "fusion")}
L = pd.read_csv("results/metrics_long.csv"); T = pd.read_csv("results/ttd.csv")
cfg = {"models": {}, "seed": SEED}
for key, (name, kind) in MODELS.items():
    thr = float(L[(L.model == name) & (L.seed == SEED)].thr.iloc[0])
    thr5 = float(T[(T.model == name) & (T.seed == SEED)].thr5.iloc[0])
    cfg["models"][key] = {"name": name, "kind": kind, "ckpt": f"checkpoints/{name}_s{SEED}.pt", "thr": thr, "thr5": thr5}
write_json(cfg, "results/demo_config.json")

P = pd.read_csv(f"results/runs/F-aug/s{SEED}/preds_test.csv")
P = P[(P.set == 1) & (P.cond == "clean")]
F = M.final_scores(P); thr = cfg["models"]["fused"]["thr"]
F["ok"] = (F.score >= thr) == (F.label == 1)
rng = np.random.default_rng(1)
os.makedirs("data/demo_samples", exist_ok=True)
fraud_ok = F[(F.label == 1) & F.ok]; norm_ok = F[(F.label == 0) & F.ok]; fraud_bad = F[(F.label == 1) & ~F.ok]
hard = fraud_bad.sort_values("score").iloc[:1] if len(fraud_bad) else F[F.label == 1].assign(m=lambda d: (d.score - thr).abs()).sort_values("m").iloc[:1]
pick = [("fraud", r) for _, r in fraud_ok.sample(2, random_state=1).iterrows()] + [("fraud-hard", r) for _, r in hard.iterrows()] + \
       [("normal", r) for _, r in norm_ok.sample(3, random_state=1).iterrows()]
out = []
for i, (kind, r) in enumerate(pick):
    dst = f"data/demo_samples/sample_{i+1}.wav"; shutil.copy(f"data/audio/set1/clean/{r.cid}.wav", dst)
    note = {"fraud": "fraud call the fused model scores correctly", "fraud-hard": "hard case: fraud call near or below the decision threshold",
            "normal": "normal call the fused model scores correctly"}[kind]
    out.append({"id": f"sample_{i+1}", "cid": r.cid, "wav": dst, "label": int(r.label), "note": note})
write_json(out, "results/demo_samples.json")
print(json.dumps(cfg, indent=1)); print([(o["id"], o["label"], o["note"]) for o in out])
