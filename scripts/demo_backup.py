"""Backup for the demo: score the 6 prepared samples with Pipeline B and record outputs + timing.  python -m scripts.demo_backup"""
import json, time
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from src.demo.scoring import Scorer, first_alert
sc = Scorer(); samples = json.load(open("results/demo_samples.json", encoding="utf8")); out = []
for s in samples:
    x, _ = sf.read(s["wav"], dtype="float32"); a8 = resample_poly(x, 1, 2).astype(np.float32)
    t = time.time(); r = sc.score(a8); el = time.time() - t
    f = r["scores"]["fused"]; tx = r["scores"]["text"]
    out.append({"id": s["id"], "label": s["label"], "note": s["note"], "audio_seconds": round(len(a8) / 8000, 1), "processing_seconds": round(el, 1),
                "times": r["times"], "fused": [round(v, 3) for v in f], "text": [round(v, 3) for v in tx], "audio": [round(v, 3) for v in r["scores"]["audio"]],
                "fused_verdict": int(f[-1] >= r["final_thr"]["fused"]), "text_verdict": int(tx[-1] >= r["final_thr"]["text"]),
                "fused_first_alert_s": first_alert(r["times"], f, r["alert_thr"]["fused"]), "text_first_alert_s": first_alert(r["times"], tx, r["alert_thr"]["text"]),
                "final_transcript": r["transcripts"][-1][:300]})
    print(s["id"], s["label"], "fused", out[-1]["fused_verdict"], "text", out[-1]["text_verdict"], "alert", out[-1]["fused_first_alert_s"], f"{el:.1f}s", flush=True)
json.dump(out, open("results/demo_backup.json", "w"), indent=1, ensure_ascii=False)
