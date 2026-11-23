"""Step-level dataset: one sample = (call variant, 10-second step k).

Text input  : last 256 subword tokens of the transcript so far (clean or recognised, per config).
Audio input : log-mel of the last 30 s of audio up to step time t (left-padded with silence).
Label       : the call's label (the model must flag fraud as early as it can).
"""
import json, os
import numpy as np
import torch
from src.features.features import window_mel, mel_cache_path


class StepData:
    def __init__(self, cfg, rows, source="asr", need_audio=True, need_text=True, audio_window=None, max_tokens=None):
        self.cfg, self.rows, self.source = cfg, rows, source
        self.need_audio, self.need_text = need_audio, need_text
        self.window = audio_window or cfg["features"]["audio_window_s"]
        self.max_tokens = max_tokens or cfg["features"]["text_tokens"]
        if need_text:
            self.vocab = json.load(open("data/features/vocab.json", encoding="utf8"))
            self.pieces = json.load(open("data/features/pieces.json", encoding="utf8"))
        self.index = [(i, k) for i, r in enumerate(rows) for k in range(len(r["steps"]))]
        self._mel = {}

    def __len__(self):
        return len(self.index)

    def ids(self, i, k):
        r = self.rows[i]; p = self.pieces[f"{r['tts_set']}|{r['cond']}|{r['cid']}"][k][self.source]
        ids = [self.vocab.get(x, 1) for x in p][-self.max_tokens:]
        return ids or [1]

    def length(self, n):
        i, k = self.index[n]
        return len(self.ids(i, k)) if self.need_text else 0

    def mel(self, i):
        m = self._mel.get(i)
        if m is None:
            m = np.load(mel_cache_path(self.cfg, self.rows[i]), mmap_mode="r"); self._mel[i] = m
        return m

    def meta(self, n):
        i, k = self.index[n]; r = self.rows[i]
        return {"set": r["tts_set"], "cond": r["cond"], "cid": r["cid"], "step": k, "t": r["steps"][k]["t"],
                "label": r["label"], "split": r["split"]}

    def get(self, n):
        i, k = self.index[n]; r = self.rows[i]; out = {"y": float(r["label"])}
        if self.need_text:
            out["ids"] = self.ids(i, k)
        if self.need_audio:
            out["mel"] = window_mel(self.mel(i), r["steps"][k]["t"], self.window)
        return out

    def batch(self, idxs):
        items = [self.get(n) for n in idxs]; b = {"y": torch.tensor([it["y"] for it in items])}
        if self.need_text:
            L = max(len(it["ids"]) for it in items)
            b["ids"] = torch.tensor([it["ids"] + [0] * (L - len(it["ids"])) for it in items])
            b["len"] = torch.tensor([len(it["ids"]) for it in items])
        if self.need_audio:
            b["mel"] = torch.from_numpy(np.stack([it["mel"] for it in items]))
        return b
