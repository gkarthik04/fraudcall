"""Train one model (text | audio | fusion) from an experiment YAML and save predictions on val and test.

python -m src.train.train --config configs/exp_T-asr.yaml --seed 0

Everything is selected on the validation split only: early stopping uses validation F1 (threshold 0.5,
step level). The test split is predicted once, after training, and never used for any choice.
Outputs: results/runs/{name}/s{seed}/{log.csv, summary.json, preds_val.csv, preds_test.csv}, checkpoints/{name}_s{seed}.pt
"""
import argparse, csv, json, os, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from src.common import load_config, seed_everything, write_json
from src.features.cache import load_manifest
from src.models.branches import TextBiLSTM, AudioCNNBiGRU, Fusion, count_params
from src.train.dataset import StepData


def select_rows(rows, split, conds, tts_set=1):
    return [r for r in rows if r["split"] == split and r["tts_set"] == tts_set and (conds is None or r["cond"] in conds)]


def bucketed_batches(data, order, bs, rng, by_length):
    """Batches of similar transcript length (LSTM cost grows with the longest sequence in a batch)."""
    order = order.tolist()
    if not by_length:
        return [order[i:i + bs] for i in range(0, len(order), bs)]
    out = []
    for c in range(0, len(order), bs * 8):
        chunk = sorted(order[c:c + bs * 8], key=data.length)
        out += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
    rng.shuffle(out)
    return out


def build_model(exp, vocab_size):
    kind = exp["kind"]
    if kind == "text":
        return TextBiLSTM(vocab_size)
    if kind == "audio":
        return AudioCNNBiGRU()
    return Fusion(TextBiLSTM(vocab_size), AudioCNNBiGRU(), mod_drop=exp.get("modality_dropout", 0.15))


def forward(model, kind, b):
    if kind == "text":
        return model(b["ids"], b["len"])[0]
    if kind == "audio":
        return model(b["mel"])[0]
    return model(b["ids"], b["len"], b["mel"])


@torch.no_grad()
def predict(model, kind, data, bs=64):
    model.eval(); out = []
    for s in range(0, len(data), bs):
        idx = list(range(s, min(s + bs, len(data))))
        out.append(torch.sigmoid(forward(model, kind, data.batch(idx))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def save_preds(path, data, scores):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "cond", "cid", "step", "t", "label", "score"])
        for n in range(len(data)):
            m = data.meta(n); w.writerow([m["set"], m["cond"], m["cid"], m["step"], m["t"], m["label"], round(float(scores[n]), 5)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3); a = ap.parse_args()
    cfg = load_config(a.config); exp = cfg["exp"]; name = exp["name"]
    torch.set_num_threads(a.threads); seed_everything(a.seed)
    rows = load_manifest(cfg); kind = exp["kind"]
    tconds = None if exp["train_conds"] == "all" else exp["train_conds"]
    tr = StepData(cfg, select_rows(rows, "train", tconds), exp.get("text_source", "asr"), kind != "text", kind != "audio",
                  exp.get("audio_window"), exp.get("text_tokens"))
    va = StepData(cfg, select_rows(rows, "val", tconds), exp.get("text_source", "asr"), kind != "text", kind != "audio",
                  exp.get("audio_window"), exp.get("text_tokens"))
    vocab = len(json.load(open("data/features/vocab.json", encoding="utf8"))) if kind != "audio" else 2
    model = build_model(exp, vocab)
    for key in ("init_text", "init_audio"):       # optional warm start of fusion branches from trained branch models
        if exp.get(key):
            sd = torch.load(f"checkpoints/{exp[key]}_s{a.seed}.pt", map_location="cpu")
            getattr(model, key.split("_")[1]).load_state_dict(sd, strict=False)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    run_dir = f"results/runs/{name}/s{a.seed}"; os.makedirs(run_dir, exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/{name}_s{a.seed}.pt"
    best, bad, t_start = -1, 0, time.time(); rng = np.random.default_rng(a.seed)
    logf = open(f"{run_dir}/log.csv", "w", newline=""); lw = csv.writer(logf); lw.writerow(["epoch", "train_loss", "val_f1", "val_loss", "sec"])
    bs = exp.get("batch_size", 32); cap = exp.get("max_samples_per_epoch", 4000)
    print(f"{name} seed{a.seed}: train samples {len(tr)}, val {len(va)}, params {params}", flush=True)
    for ep in range(exp["epochs"]):
        model.train(); t0 = time.time()
        order = rng.permutation(len(tr))[:cap]; tot, nb = 0.0, 0
        batches = bucketed_batches(tr, order, bs, rng, kind != "audio")
        for idx in batches:
            b = tr.batch(idx)
            loss = lossf(forward(model, kind, b), b["y"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sc = predict(model, kind, va); y = np.array([va.meta(n)["label"] for n in range(len(va))])
        f1 = f1_score(y, sc > 0.5); vl = float(nn.functional.binary_cross_entropy(torch.tensor(sc).clamp(1e-6, 1 - 1e-6), torch.tensor(y, dtype=torch.float32)))
        lw.writerow([ep, round(tot / nb, 4), round(f1, 4), round(vl, 4), round(time.time() - t0, 1)]); logf.flush()
        print(f"  ep{ep} loss {tot/nb:.3f} val_f1 {f1:.3f} val_loss {vl:.3f} {time.time()-t0:.0f}s", flush=True)
        if f1 > best:
            best, bad = f1, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= exp.get("patience", 3):
                break
    model.load_state_dict(torch.load(ckpt))
    # validation predictions (all val conditions) and test predictions (all conditions, both TTS sets)
    vall = StepData(cfg, [r for r in rows if r["split"] == "val" and r["tts_set"] == 1], exp.get("text_source", "asr"), kind != "text", kind != "audio", exp.get("audio_window"), exp.get("text_tokens"))
    save_preds(f"{run_dir}/preds_val.csv", vall, predict(model, kind, vall))
    te = StepData(cfg, [r for r in rows if r["split"] == "test"], exp.get("text_source", "asr"), kind != "text", kind != "audio", exp.get("audio_window"), exp.get("text_tokens"))
    save_preds(f"{run_dir}/preds_test.csv", te, predict(model, kind, te))
    write_json({"name": name, "seed": a.seed, "params": params, "best_val_f1": best, "epochs_run": ep + 1,
                "train_seconds": round(time.time() - t_start, 1), "train_samples": len(tr), "exp": exp}, f"{run_dir}/summary.json")
    print("done", name, a.seed, "best val f1", round(best, 4), flush=True)


if __name__ == "__main__":
    main()
