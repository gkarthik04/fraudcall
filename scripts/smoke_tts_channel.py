import json, time, numpy as np, soundfile as sf
from src.common import load_config
from src.tts.synth import make_engine, assemble_call
from src.channel.channel import apply_channel, TEST_GRID

cfg = load_config(); convs = json.load(open("data/processed/conversations.json"))
eng = make_engine("piper", cfg); t = time.time()
a, meta = assemble_call(convs[0], eng, cfg)
print("synth", round(time.time() - t, 1), "s for", meta["duration"], "s audio; turns", len(meta["turns"]), meta["voices"], convs[0]["pace"], convs[0]["label"])
for k in ["hard", "gsm", "snr5"]:
    t = time.time(); y = apply_channel(a, TEST_GRID[k], 7); y2 = apply_channel(a, TEST_GRID[k], 7)
    print(k, len(y) / 8000, "deterministic:", bool(np.array_equal(y, y2)), "rms", float(np.sqrt((y ** 2).mean())), round(time.time() - t, 2))
sf.write("tmp_phase0/smoke_clean.wav", a, 16000)
sf.write("tmp_phase0/smoke_hard.wav", apply_channel(a, TEST_GRID["hard"], 7), 8000)
