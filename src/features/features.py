"""Feature extraction: log-mel spectrograms (audio branch) and subword tokens (text branch).

Log-mel: 8 kHz audio, 25 ms window (400 samples), 20 ms hop (50 frames/s), 64 mel bins up to 4 kHz.
Values are log(mel + 1e-5) then shifted/scaled by a fixed constant, so silence is about -2.2.
Tokens: the XLM-RoBERTa sentencepiece tokenizer (multilingual subwords), remapped to a compact vocabulary
built from training transcripts, because the full 250k vocabulary would make the embedding table huge on CPU.
"""
import os, json, collections
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.signal import resample_poly

SR, N_FFT, HOP, N_MELS = 8000, 400, 160, 64
FPS = SR // HOP                  # 50 frames per second
_FB = torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=60, fmax=4000), dtype=torch.float32)
_WIN = torch.hann_window(N_FFT)
SILENCE = (np.log(1e-5) + 6.0) / 2.5


def load_8k(path):
    """Load a call as 8 kHz float32 (clean 16 kHz references are down-sampled, no codec)."""
    x, sr = sf.read(path, dtype="float32")
    if sr == 16000:
        x = resample_poly(x, 1, 2).astype(np.float32)
    return x


def logmel(x8k):
    """(n_samples,) float32 at 8 kHz -> (64, frames) float32, normalised."""
    t = torch.from_numpy(np.ascontiguousarray(x8k))
    S = torch.stft(t, N_FFT, HOP, window=_WIN, center=True, return_complex=True).abs() ** 2
    m = torch.log(_FB @ S + 1e-5)
    return ((m + 6.0) / 2.5).numpy().astype(np.float32)


def mel_cache_path(cfg, row):
    return os.path.join("data/features", f"set{row['tts_set']}", row["cond"], row["cid"] + ".npy")


def cache_mel(cfg, row):
    p = mel_cache_path(cfg, row)
    if os.path.exists(p):
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, logmel(load_8k(row["audio"])).astype(np.float16))
    return p


def window_mel(mel, t, seconds=30):
    """Last `seconds` of mel (64, T) ending at time t; left-padded with the silence value."""
    n = int(seconds * FPS); end = min(int(round(t * FPS)), mel.shape[1]); start = max(0, end - n)
    w = mel[:, start:end].astype(np.float32)
    if w.shape[1] < n:
        w = np.concatenate([np.full((N_MELS, n - w.shape[1]), SILENCE, np.float32), w], axis=1)
    return w


# ------------------------------------------------------------------ text
class Tokenizer:
    """XLM-R subword tokenizer + compact vocab. id 0 = pad, 1 = unk."""
    def __init__(self, vocab=None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1}

    def pieces(self, text):
        return self.tok.tokenize(text) if text else []

    def fit(self, texts, min_freq=2):
        c = collections.Counter(p for t in texts for p in self.pieces(t))
        for p, n in c.most_common():
            if n >= min_freq and p not in self.vocab:
                self.vocab[p] = len(self.vocab)
        return self

    def encode(self, text, max_len=256):
        ids = [self.vocab.get(p, 1) for p in self.pieces(text)]
        return ids[-max_len:] if ids else [1]   # keep the LAST max_len tokens (most recent speech)

    def save(self, path):
        json.dump(self.vocab, open(path, "w", encoding="utf8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf8")))
