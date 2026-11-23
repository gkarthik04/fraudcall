import json, numpy as np
from sklearn.metrics import roc_auc_score
rows=[json.loads(l) for l in open("data/manifest/manifest.jsonl",encoding="utf8")]
c=[r for r in rows if r["tts_set"]==1 and r["cond"]=="clean" and r["split"]!="test"]
y=np.array([r["label"] for r in c]); 
pace=np.array([{"fast":1,"normal":0,"slow":-1}[r["pace"]] for r in c])
print("n",len(c),"fraud rate by pace",{p:round(float(y[pace==v].mean()),3) for p,v in (("fast",1),("normal",0),("slow",-1))}, "AUROC pace", round(roc_auc_score(y,pace),3))
wps=np.array([len(r["clean_transcript"].split())/r["duration"] for r in c]); print("AUROC words/sec",round(roc_auc_score(y,wps),3))
# agent speaking rate measured from the rendered audio (words per second of agent speech)
import glob
rate = []
for r in c:
    m = json.load(open(f"data/audio/set1/meta/{r['cid']}.json", encoding="utf8"))
    ag = [t for t in m["turns"] if t["speaker"] == "Agent" and not t["truncated"]]
    w = sum(len(t["spoken"].split()) for t in ag); d = sum(t["end"] - t["start"] for t in ag)
    rate.append(w / max(d, 1e-3))
rate = np.array(rate); print("AUROC agent words/sec", round(roc_auc_score(y, rate), 3), "mean fraud", rate[y==1].mean().round(2), "normal", rate[y==0].mean().round(2), "sd", rate.std().round(2))
json.dump({"n_calls_train_val_clean": len(c), "fraud_rate_by_pace_label": {p: float(y[pace == v].mean()) for p, v in (("fast", 1), ("normal", 0), ("slow", -1))},
           "auroc_pace_label_from_metadata": float(roc_auc_score(y, pace)), "auroc_words_per_sec_whole_call": float(roc_auc_score(y, wps)),
           "auroc_agent_speech_rate_measured_in_audio": float(roc_auc_score(y, rate)), "agent_rate_mean_sd": [float(rate.mean()), float(rate.std())]},
          open("results/pace_diagnostic.json", "w"), indent=1)
