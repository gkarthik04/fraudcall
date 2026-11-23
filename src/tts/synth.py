"""Text-to-speech wrappers (Piper = Set 1, MMS-TTS = Set 2) and call assembly.

A "call" is the turns of one ICFD conversation, each rendered by a TTS voice (one voice per speaker for
the whole call) and joined with short silences, capped at call.max_seconds. We also record, for every
turn, its start/end time so that the *clean* transcript at any time t can be reconstructed (words are
assumed evenly spread inside a turn).
"""
import re, hashlib, os
import numpy as np
from scipy.signal import resample_poly

SR = 16000
RUPEE = "₹"


def tts_normalize(text):
    """Make written text speakable: currency, symbols, emoji."""
    t = re.sub(RUPEE + r"\s?([\d,\.]+)", r"\1 rupees", text)
    t = re.sub(r"\bRs\.?\s?([\d,\.]+)", r"\1 rupees", t)
    t = re.sub(r"[*_#~`<>\[\]{}|\\]", " ", t)
    t = re.sub(r"[^\x00-\x7Fऀ-ॿ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _rs(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _seed(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2 ** 32)


class PiperEngine:
    """Piper VITS voices; libritts_r has 904 built-in speakers. Fast on CPU (3-14x real time)."""
    name = "piper"

    def __init__(self, cfg):
        from piper import PiperVoice
        d = cfg["paths"]["piper_dir"]; m = cfg["tts"]["set1"]["model"]
        self.voice = PiperVoice.load(os.path.join(d, m + ".onnx"))
        try:   # limit ONNX threads so several worker processes do not oversubscribe the CPU
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
            self.voice.session = ort.InferenceSession(os.path.join(d, m + ".onnx"), sess_options=so,
                                                      providers=["CPUExecutionProvider"])
        except Exception:
            pass
        n_spk = self.voice.config.num_speakers
        rng = np.random.default_rng(cfg["seed"])
        self.pool = [int(x) for x in rng.choice(n_spk, size=cfg["tts"]["set1"]["n_voices"], replace=False)]
        self.sr = self.voice.config.sample_rate

    def voices_for(self, cid):
        rng = np.random.default_rng(_seed(cid))
        a, b = rng.choice(len(self.pool), size=2, replace=False)
        return {"Agent": self.pool[a], "Customer": self.pool[b]}

    def say(self, text, voice, length_scale=1.0, noise_scale=None):
        from piper import SynthesisConfig
        cfg = SynthesisConfig(speaker_id=voice, length_scale=length_scale, noise_scale=noise_scale)
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text, cfg)]
        if not chunks:
            return np.zeros(int(0.2 * SR), np.float32)
        return _rs(np.concatenate(chunks).astype(np.float32) / 32768, self.sr)


class MMSEngine:
    """Meta MMS-TTS (VITS, single voice). Slow on CPU (~0.8x real time). Used only for the unseen test set.
    Because it has one voice, 'voices' are variants: pitch/tempo (by resampling) and speaking rate."""
    name = "mms"

    def __init__(self, cfg):
        import torch
        from transformers import VitsModel, AutoTokenizer
        torch.set_num_threads(cfg["asr"]["threads"])
        repo = cfg["tts"]["set2"]["model"]
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = VitsModel.from_pretrained(repo).eval()
        self.sr = self.model.config.sampling_rate
        self.variants = [0.92, 0.96, 1.0, 1.05, 1.1]

    def voices_for(self, cid):
        rng = np.random.default_rng(_seed("mms" + cid))
        a, b = rng.choice(len(self.variants), size=2, replace=False)
        return {"Agent": int(a), "Customer": int(b)}

    def say(self, text, voice, length_scale=1.0, noise_scale=None):
        import torch
        inp = self.tok(text, return_tensors="pt")
        if inp["input_ids"].shape[1] == 0:
            return np.zeros(int(0.2 * SR), np.float32)
        self.model.speaking_rate = 1.0 / length_scale
        torch.manual_seed(_seed(text) % 100000)
        with torch.no_grad():
            w = self.model(**inp).waveform[0].numpy().astype(np.float32)
        f = self.variants[voice]          # resample trick: changes pitch and tempo together
        w = resample_poly(w, 1000, int(round(1000 * f))).astype(np.float32)
        return _rs(w, self.sr)


def make_engine(name, cfg):
    return PiperEngine(cfg) if name == "piper" else MMSEngine(cfg)


def assemble_call(conv, engine, cfg, seed=0):
    """Synthesize all turns of a conversation. Returns (audio16k float32, meta dict)."""
    rng = np.random.default_rng(_seed(conv["cid"]) + seed)
    cap = cfg["call"]["max_seconds"]; gmin, gmax = cfg["call"]["turn_gap_seconds"]
    voices = engine.voices_for(conv["cid"])
    pace_ls = cfg["tts"]["pace_length_scale"][conv["pace"]]
    hot = any(w in conv["emotion"] for w in ("urgen", "anger", "angry", "fear", "aggress", "threat"))
    segs, turns, t = [], [], 0.0
    for tr in conv["turns"]:
        if t >= cap:
            break
        text = tts_normalize(tr["text"])
        if not text:
            continue
        spk = "Agent" if tr["speaker"].lower().startswith("agent") else "Customer"
        is_agent = spk == "Agent"
        ns = (0.9 if hot else 0.667) if (is_agent and engine.name == "piper") else None
        a = engine.say(text, voices[spk], length_scale=pace_ls if is_agent else 1.0, noise_scale=ns)
        start = t; end = min(t + len(a) / SR, cap)
        full = len(a) / SR
        a = a[: int((end - start) * SR)]
        segs.append(a)
        gap = float(rng.uniform(gmin, gmax)); segs.append(np.zeros(int(gap * SR), np.float32))
        turns.append({"speaker": spk, "start": round(start, 3), "end": round(end, 3), "text": tr["text"],
                      "spoken": text, "voice": voices[spk], "full": round(full, 3), "truncated": bool(end - start < full - 1e-3)})
        t = end + gap
    audio = np.concatenate(segs)[: int(cap * SR)] if segs else np.zeros(SR, np.float32)
    meta = {"cid": conv["cid"], "engine": engine.name, "duration": round(len(audio) / SR, 3), "turns": turns,
            "voices": voices, "pace_length_scale": pace_ls}
    return audio, meta


def clean_words_until(meta, t):
    """Clean (reference) transcript up to time t; words are evenly spread inside each turn."""
    out = []
    for tr in meta["turns"]:
        words = tr["spoken"].split()
        if not words or tr["start"] >= t:
            continue
        dur = max(tr.get("full", tr["end"] - tr["start"]), 1e-6)   # words spread over the untruncated turn
        n = len(words)
        for i, w in enumerate(words):
            wt = tr["start"] + dur * (i + 1) / n
            if wt <= t:
                out.append(w)
    return " ".join(out)
