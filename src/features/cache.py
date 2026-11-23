"""Cache features for every manifest row: log-mel .npy files and subword pieces of each step's transcript.

Run:  python -m src.features.cache
Writes data/features/set{N}/{cond}/{cid}.npy, data/features/pieces.json, data/features/vocab.json
(the vocabulary is fitted on TRAIN rows only).
"""
import json, os, collections
from src.common import load_config
from src.features.features import cache_mel, Tokenizer


def load_manifest(cfg):
    with open(cfg["paths"]["manifest"] + "/manifest.jsonl", encoding="utf8") as f:
        return [json.loads(l) for l in f]


def main():
    cfg = load_config(); rows = load_manifest(cfg)
    for i, r in enumerate(rows):
        cache_mel(cfg, r)
        if i % 500 == 0:
            print("mel", i, "/", len(rows), flush=True)
    tok = Tokenizer(); pieces = {}; cnt = collections.Counter()
    for r in rows:
        key = f"{r['tts_set']}|{r['cond']}|{r['cid']}"
        pieces[key] = [{"clean": tok.pieces(s["clean"]), "asr": tok.pieces(s["asr"])} for s in r["steps"]]
        if r["split"] == "train" and r["tts_set"] == 1:
            last = pieces[key][-1]
            cnt.update(last["clean"]); cnt.update(last["asr"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for p, n in cnt.most_common():
        if n >= 2:
            vocab[p] = len(vocab)
    json.dump(pieces, open("data/features/pieces.json", "w", encoding="utf8"), ensure_ascii=False)
    json.dump(vocab, open("data/features/vocab.json", "w", encoding="utf8"), ensure_ascii=False)
    print("vocab size", len(vocab))


if __name__ == "__main__":
    main()
