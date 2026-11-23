"""Phase 0: benchmark TTS engines (Piper, MMS-TTS) on CPU and ASR (faster-whisper) on the output.

Writes results/phase0_tts_asr.json and a few wavs under tmp_phase0/ (untracked).
"""
import json, time, io, glob, os, sys
import numpy as np, soundfile as sf, zstandard, jiwer, torch
from scipy.signal import resample_poly

OUT = "tmp_phase0"; os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)

def load_rows(n=400):
    p = sorted(glob.glob("data/raw/icfd/source_conversations/train-*.zst"))[0]
    rows = []
    with open(p, "rb") as f:
        for l in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf8"):
            rows.append(json.loads(l))
            if len(rows) >= n: break
    return rows

rows = load_rows()
hing = next(r for r in rows if r["scenario"]["case_type"] == "Clear Fraud")
turns_hing = [(t["speaker"], t["text"]) for t in hing["transcript"][:5]]
turns_en = [(t["speaker"], t["text"]) for t in
            next(r for r in rows if r["scenario"]["case_type"] == "Clear Normal")["transcript"][:5]]
turns_hi = [("Agent", "नमस्ते, मैं आपके बैंक से बोल रहा हूँ। आपके खाते में एक संदिग्ध लेनदेन हुआ है।"),
            ("Customer", "अच्छा, कितने रुपये का लेनदेन है?"),
            ("Agent", "पचास हज़ार रुपये का। कृपया अपना ओटीपी बताइए ताकि हम इसे रोक सकें।"),
            ("Customer", "मैं ओटीपी नहीं बता सकता, मैं बैंक जाकर पता करूँगा।")]

results = {"tts": [], "asr": []}

# ---------------- Piper ----------------
from piper import PiperVoice, SynthesisConfig
def piper_synth(voice, text, sid=None):
    cfg = SynthesisConfig(speaker_id=sid)
    chunks = [c.audio_int16_array for c in voice.synthesize(text, cfg)]
    return np.concatenate(chunks).astype(np.float32) / 32768, voice.config.sample_rate

piper_models = {
  "piper_en_libritts_r": ("data/models/piper/en_US-libritts_r-medium.onnx", [(12, 40)], ["en_hing", "en"]),
  "piper_hi_pratham": ("data/models/piper/hi_IN-pratham-medium.onnx", [(None, None)], ["hi"]),
}
convs = {"en_hing": turns_hing, "en": turns_en, "hi": turns_hi}

def run_conv(name, synth_fn, conv_key, spk_cfg):
    t0 = time.time(); segs = []; sr = None
    for i, (spk, text) in enumerate(convs[conv_key]):
        a, sr = synth_fn(text, spk_cfg[0 if spk == "Agent" else 1])
        segs.append(a); segs.append(np.zeros(int(0.3 * sr), np.float32))
    dt = time.time() - t0; audio = np.concatenate(segs); dur = len(audio) / sr
    path = f"{OUT}/{name}_{conv_key}.wav"; sf.write(path, audio, sr)
    r = {"model": name, "conv": conv_key, "audio_sec": round(dur, 2), "compute_sec": round(dt, 2), "rtf_audio_per_compute": round(dur / dt, 2), "sr": sr, "wav": path}
    results["tts"].append(r); print(r, flush=True)

for name, (path, spk, keys) in piper_models.items():
    voice = PiperVoice.load(path)
    for k in keys:
        if name.startswith("piper_en"):
            run_conv(name, lambda t, s: piper_synth(voice, t, s), k, (12, 40))
        else:
            run_conv(name, lambda t, s: piper_synth(voice, t, None), k, (None, None))

# ---------------- MMS-TTS ----------------
from transformers import VitsModel, AutoTokenizer
for name, repo, keys in [("mms_eng", "facebook/mms-tts-eng", ["en_hing", "en"]), ("mms_hin", "facebook/mms-tts-hin", ["hi"])]:
    try:
        tok = AutoTokenizer.from_pretrained(repo); m = VitsModel.from_pretrained(repo).eval()
        def synth(text, _s, tok=tok, m=m):
            inp = tok(text, return_tensors="pt")
            if inp["input_ids"].shape[1] == 0: return np.zeros(1600, np.float32), m.config.sampling_rate
            with torch.no_grad(): w = m(**inp).waveform[0].numpy()
            return w.astype(np.float32), m.config.sampling_rate
        for k in keys: run_conv(name, synth, k, (None, None))
    except Exception as e:
        print("MMS fail", name, repr(e)); results["tts"].append({"model": name, "error": repr(e)})

# ---------------- ASR ----------------
from faster_whisper import WhisperModel
refs = {k: " ".join(t for _, t in v) for k, v in convs.items()}
norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
for size in ["tiny", "base"]:
    wm = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=6)
    for r in [x for x in results["tts"] if "wav" in x]:
        a, sr = sf.read(r["wav"]); a16 = resample_poly(a, 16000, sr).astype(np.float32)
        lang = "hi" if r["conv"] == "hi" else "en"
        t0 = time.time(); segs, _ = wm.transcribe(a16, language=lang, beam_size=1); hyp = " ".join(s.text for s in segs); dt = time.time() - t0
        wer = jiwer.wer(norm(refs[r["conv"]]), norm(hyp)) if lang == "en" else jiwer.wer(refs[r["conv"]].replace("।", ""), hyp.replace("।", "").replace(",", "").replace("?", ""))
        rr = {"asr": size, "tts": r["model"], "conv": r["conv"], "wer": round(wer, 3), "asr_sec": round(dt, 2), "asr_rtf_x": round(r["audio_sec"] / dt, 2), "hyp": hyp[:200]}
        results["asr"].append(rr); print(rr, flush=True)

json.dump(results, open("results/phase0_tts_asr.json", "w"), indent=1, ensure_ascii=False)
