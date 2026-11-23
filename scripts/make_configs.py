"""Write one YAML per experiment into configs/ (run: python -m scripts.make_configs)."""
import yaml
C, A3 = ["clean"], ["clean", "rand0", "rand1"]
E = {
 "T-clean":   dict(kind="text",   text_source="clean", train_conds=C,  lr=2e-3, epochs=8),
 "T-asr":     dict(kind="text",   text_source="asr",   train_conds=C,  lr=2e-3, epochs=8),
 "T-asr-aug": dict(kind="text",   text_source="asr",   train_conds=A3, lr=2e-3, epochs=8),
 "A":         dict(kind="audio",  train_conds=C,  lr=1e-3, epochs=20, patience=6),
 "A-aug":     dict(kind="audio",  train_conds=A3, lr=1e-3, epochs=20, patience=6),
 "F":         dict(kind="fusion", text_source="asr", train_conds=C,  lr=5e-4, epochs=5, init_text="T-asr", init_audio="A"),
 "F-aug":     dict(kind="fusion", text_source="asr", train_conds=A3, lr=5e-4, epochs=5, init_text="T-asr-aug", init_audio="A-aug"),
 # ablations
 "F-aug-win10": dict(kind="fusion", text_source="asr", train_conds=A3, lr=5e-4, epochs=5, audio_window=10, init_text="T-asr-aug", init_audio="A-aug"),
 "F-aug-win60": dict(kind="fusion", text_source="asr", train_conds=A3, lr=5e-4, epochs=5, audio_window=60, init_text="T-asr-aug", init_audio="A-aug"),
 "T-asr-tok512": dict(kind="text", text_source="asr", train_conds=C, lr=2e-3, epochs=8, text_tokens=512),
 "F-aug-tok128": dict(kind="fusion", text_source="asr", train_conds=A3, lr=5e-4, epochs=5, text_tokens=128, init_text="T-asr-aug", init_audio="A-aug"),
}
for n, e in E.items():
    e = dict({"patience": 2}, **e); e.update(name=n, batch_size=32, max_samples_per_epoch=2000)
    yaml.safe_dump({"exp": e}, open(f"configs/exp_{n}.yaml", "w"), sort_keys=False)
print(len(E), "configs")
