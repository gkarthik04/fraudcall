import json, numpy as np
from src.common import load_config
from src.tts.synth import make_engine
cfg = load_config(); e = make_engine("piper", cfg)
txt = "Hello sir, I am calling from your bank about a suspicious transaction on your account."
for ls in (0.7, 0.85, 1.0, 1.15):
    print(ls, round(len(e.say(txt, 232, length_scale=ls)) / 16000, 2), "s", "| noise_scale none/0.9 ->", round(len(e.say(txt, 232, length_scale=ls, noise_scale=0.9)) / 16000, 2))
m = [json.load(open(f"data/audio/set1/meta/{c['cid']}.json")) for c in json.load(open("data/processed/conversations.json"))[:60]]
print({k: round(float(np.mean([x["pace_length_scale"] for x in m])),3) for k in ["mean_ls"]})
