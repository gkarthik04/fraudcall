"""Pipeline B: score one call in cumulative 10-second steps (used by the Streamlit demo and the backup script).

audio -> 8 kHz mono -> ASR once (word timestamps) -> for every step t: transcript so far -> text BiLSTM,
log-mel of the last 30 s -> audio CNN-BiGRU, both -> fusion.  Mirrors exactly what training saw.
Uploaded audio is processed in memory only and never written to disk by this module.
"""
import json, os, subprocess, tempfile
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from src.common import load_config
from src.asr.asr import ASR, words_until
from src.features.features import logmel, window_mel, Tokenizer
from src.models.branches import TextBiLSTM, AudioCNNBiGRU, Fusion


def decode_to_16k(raw_bytes, suffix):
    """Decode WAV/MP3 bytes to float32 mono 16 kHz via ffmpeg (temp files deleted immediately)."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in" + suffix); dst = os.path.join(d, "out.wav")
        open(src, "wb").write(raw_bytes)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)
        x, _ = sf.read(dst, dtype="float32")
    return x


class Scorer:
    def __init__(self, demo_cfg_path="results/demo_config.json"):
        self.cfg = load_config()
        self.dc = json.load(open(demo_cfg_path, encoding="utf8"))
        self.vocab = json.load(open("data/features/vocab.json", encoding="utf8")); self.tok = Tokenizer(self.vocab)
        self.asr = ASR(self.cfg); self.models = {}
        for key, spec in self.dc["models"].items():
            torch.set_num_threads(4)
            if spec["kind"] == "text":
                m = TextBiLSTM(len(self.vocab))
            elif spec["kind"] == "audio":
                m = AudioCNNBiGRU()
            else:
                m = Fusion(TextBiLSTM(len(self.vocab)), AudioCNNBiGRU(), mod_drop=0.0)
            m.load_state_dict(torch.load(spec["ckpt"], map_location="cpu")); m.eval(); self.models[key] = (m, spec)

    @torch.no_grad()
    def score(self, audio8k, on_step=None):
        """Return dict: times, transcripts, scores {model: [..]}, thresholds. on_step(t, text, scores) is called live."""
        dur = len(audio8k) / 8000; step = self.cfg["call"]["step_seconds"]
        words = self.asr.transcribe(audio8k, 8000)
        mel = logmel(audio8k)
        times = list(np.arange(step, dur, step)) + [dur]
        out = {"times": [round(float(t), 1) for t in times], "transcripts": [], "scores": {k: [] for k in self.models}}
        for t in times:
            text = words_until(words, t); out["transcripts"].append(text)
            ids = torch.tensor([self.tok.encode(text, 256)]); ln = torch.tensor([ids.shape[1]])
            m_in = torch.from_numpy(window_mel(mel, t, 30))[None]
            cur = {}
            for k, (m, spec) in self.models.items():
                if spec["kind"] == "text":
                    p = torch.sigmoid(m(ids, ln)[0])
                elif spec["kind"] == "audio":
                    p = torch.sigmoid(m(m_in)[0])
                else:
                    p = torch.sigmoid(m(ids, ln, m_in))
                cur[k] = float(p[0]); out["scores"][k].append(cur[k])
            if on_step:
                on_step(float(t), text, cur)
        out["thresholds"] = {k: spec["thr5"] for k, (m, spec) in self.models.items()}
        out["alert_thr"] = {k: spec["thr5"] for k, (m, spec) in self.models.items()}
        out["final_thr"] = {k: spec["thr"] for k, (m, spec) in self.models.items()}
        return out


def first_alert(times, scores, thr):
    for t, s in zip(times, scores):
        if s >= thr:
            return t
    return None
