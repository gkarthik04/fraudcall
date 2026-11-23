"""Phone-channel simulation: noise -> 8 kHz resample -> telephony codec -> packet loss.

apply_channel(audio, cfg, seed) is deterministic: all randomness comes from `seed` and ffmpeg is
deterministic for a fixed input. Input: float32 mono at 16 kHz. Output: float32 mono at 8 kHz
(or the unchanged 16 kHz signal for the clean reference condition cfg == {"name": "clean"}).

cfg keys (all optional): codec in {none, mulaw, alaw, gsm, opus8, opus12, amrnb};
                         snr_db (float | None); noise in {white, pink, brown}; loss_pct (0-100).
"""
import subprocess, tempfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, lfilter

CODECS = {  # name -> (ffmpeg encoder args, container extension)
    "mulaw": (["-c:a", "pcm_mulaw"], "wav"),
    "alaw": (["-c:a", "pcm_alaw"], "wav"),
    "gsm": (["-c:a", "libgsm"], "gsm"),
    "opus8": (["-c:a", "libopus", "-b:a", "8k"], "ogg"),
    "opus12": (["-c:a", "libopus", "-b:a", "12k"], "ogg"),
    "amrnb": (["-c:a", "libopencore_amrnb", "-b:a", "12.2k"], "amr"),
}


def _noise(kind, n, rng):
    w = rng.standard_normal(n)
    if kind == "pink":      # approximately 1/f noise through an IIR filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        w = lfilter(b, a, w)
    elif kind == "brown":
        w = np.cumsum(w); w = w - np.linspace(w[0], w[-1], n)
    return w / (np.std(w) + 1e-9)


def add_noise(x, snr_db, kind, rng):
    """Add generated noise so that signal power / noise power equals snr_db."""
    p_sig = np.mean(x ** 2) + 1e-12
    n = _noise(kind, len(x), rng) * np.sqrt(p_sig / (10 ** (snr_db / 10)))
    return x + n


def run_codec(x8k, codec):
    """Encode then decode through ffmpeg. x8k is float32 at 8 kHz."""
    args, ext = CODECS[codec]
    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = f"{d}/in.wav", f"{d}/enc.{ext}", f"{d}/dec.wav"
        sf.write(src, x8k, 8000, subtype="PCM_16")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "8000", "-ac", "1", *args, enc],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError(r.stderr.decode()[:300])
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-ar", "8000", "-ac", "1",
                            "-c:a", "pcm_s16le", dec], capture_output=True)
        if r.returncode:
            raise RuntimeError(r.stderr.decode()[:300])
        y, _ = sf.read(dec, dtype="float32")
    return y[: len(x8k)] if len(y) >= len(x8k) else np.pad(y, (0, len(x8k) - len(y)))


def drop_packets(x, loss_pct, rng, frame=160):
    """Zero out a fraction of 20 ms frames (160 samples at 8 kHz), chosen at random."""
    if loss_pct <= 0:
        return x
    y = x.copy(); nf = len(x) // frame
    k = int(round(nf * loss_pct / 100.0))
    for i in rng.choice(nf, size=k, replace=False):
        y[i * frame:(i + 1) * frame] = 0
    return y


def apply_channel(audio16k, cfg, seed):
    """Deterministic phone-channel degradation. See module docstring."""
    if cfg.get("name") == "clean":
        return audio16k.astype(np.float32)
    rng = np.random.default_rng(seed)
    x = audio16k.astype(np.float32)
    if cfg.get("snr_db") is not None:
        x = add_noise(x, cfg["snr_db"], cfg.get("noise", "white"), rng)
    x8 = np.clip(resample_poly(x, 1, 2).astype(np.float32), -1, 1)
    codec = cfg.get("codec", "none")
    if codec != "none":
        x8 = run_codec(x8, codec)
    x8 = drop_packets(x8, cfg.get("loss_pct", 0), rng)
    return np.clip(x8, -1, 1).astype(np.float32)


# ---------------- condition sets ----------------
TEST_GRID = {   # fixed grid used for every test-split evaluation
    "clean": {"name": "clean"},
    "bw8k": {"name": "bw8k", "codec": "none"},
    "mulaw": {"name": "mulaw", "codec": "mulaw"},
    "alaw": {"name": "alaw", "codec": "alaw"},
    "gsm": {"name": "gsm", "codec": "gsm"},
    "opus8": {"name": "opus8", "codec": "opus8"},
    "opus12": {"name": "opus12", "codec": "opus12"},
    "amrnb": {"name": "amrnb", "codec": "amrnb"},
    "snr20": {"name": "snr20", "codec": "none", "snr_db": 20, "noise": "white"},
    "snr10": {"name": "snr10", "codec": "none", "snr_db": 10, "noise": "white"},
    "snr5": {"name": "snr5", "codec": "none", "snr_db": 5, "noise": "white"},
    "loss5": {"name": "loss5", "codec": "none", "loss_pct": 5},
    "loss10": {"name": "loss10", "codec": "none", "loss_pct": 10},
    "hard": {"name": "hard", "codec": "opus8", "snr_db": 5, "noise": "white", "loss_pct": 10},
}
SET2_CONDITIONS = ["clean", "gsm", "snr10", "hard"]


def sample_train_channel(cfg, rng):
    """Random channel for training/validation: random codec, SNR 5-25 dB, loss 0-15 %."""
    c = cfg["channel"]
    d = {"name": "rand", "codec": str(rng.choice(c["train_codecs"]))}
    if rng.random() < c["train_p_noise"]:
        d["snr_db"] = float(round(rng.uniform(*c["train_snr_db"]), 1))
        d["noise"] = str(rng.choice(["white", "pink", "brown"]))
    if rng.random() < c["train_p_loss"]:
        d["loss_pct"] = float(round(rng.uniform(*c["train_loss_pct"]), 1))
    return d
