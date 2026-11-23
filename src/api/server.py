"""Backend for the web frontend.   python -m uvicorn src.api.server:app --port 8000   then open http://localhost:8000

Serves the static page in web/ and a small JSON API:
  GET  /api/health          models loaded yet?
  GET  /api/samples         held-out demo calls (labels stay hidden until scoring is done)
  GET  /api/sample/{id}.wav audio of a demo call (synthetic speech)
  POST /api/score           stream (NDJSON) of live scoring steps, then the final result. Uploaded audio is decoded in memory
                            (a temp folder deleted at once) and never stored.
  GET  /api/results         tables read from the logged runs under results/
"""
import asyncio, json, os, queue, subprocess, threading, time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly

from src.channel.channel import TEST_GRID, apply_channel
from src.demo.scoring import Scorer, decode_to_16k, first_alert

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)                                   # the pipeline uses paths relative to the project root
CHANNELS = {"clean": "clean (no degradation)", "mulaw": "G.711 mu-law", "gsm": "GSM", "opus8": "Opus 8 kbps",
            "snr10": "noisy (SNR 10 dB)", "loss10": "lossy (10 % packets lost)"}
MAX_UPLOAD = 25 * 1024 * 1024
STATE = {"scorer": None, "error": None}
BUSY = threading.Lock()


def _load():
    try:
        STATE["scorer"] = Scorer()
    except Exception as e:                       # surfaced through /api/health
        STATE["error"] = repr(e)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="FraudCall", lifespan=lifespan)


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"          # always revalidate the page, scripts and styles while developing
    return resp


def samples():
    return json.load(open("results/demo_samples.json", encoding="utf8"))


def to_8k(audio16k, key):
    if key == "clean":
        return resample_poly(audio16k, 1, 2).astype(np.float32)
    return apply_channel(audio16k, TEST_GRID[key], seed=0)


def verdict(score, thr):
    flag = score >= thr
    return ("FRAUD LIKELY" if flag else "No fraud detected"), float(score if flag else 1 - score)


@app.get("/api/health")
def health():
    return {"ready": STATE["scorer"] is not None, "error": STATE["error"]}


@app.get("/api/config")
def config():
    return {"channels": [{"key": k, "label": v} for k, v in CHANNELS.items()],
            "samples": [{"id": s["id"], "url": f"/api/sample/{s['id']}.wav"} for s in samples()]}


@app.get("/api/sample/{sid}.wav")
def sample_audio(sid: str):
    s = next((x for x in samples() if x["id"] == sid), None)
    if s is None:
        raise HTTPException(404, "unknown sample")
    return FileResponse(s["wav"], media_type="audio/wav")


@app.post("/api/score")
async def score(source: str = Form(...), channel: str = Form("clean"), sample_id: str = Form(None), file: UploadFile = File(None)):
    sc = STATE["scorer"]
    if sc is None:
        raise HTTPException(503, "models are still loading")
    if channel not in CHANNELS:
        raise HTTPException(400, "unknown channel")
    truth = None
    if source == "sample":
        s = next((x for x in samples() if x["id"] == sample_id), None)
        if s is None:
            raise HTTPException(400, "unknown sample")
        audio16, _ = sf.read(s["wav"], dtype="float32"); truth = s
    elif source == "upload":
        if file is None:
            raise HTTPException(400, "no file")
        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in (".wav", ".mp3"):
            raise HTTPException(400, "only WAV or MP3")
        raw = await file.read()
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "file larger than 25 MB")
        try:
            audio16 = decode_to_16k(raw, suffix)        # in memory / temp dir deleted at once
        except (subprocess.CalledProcessError, RuntimeError, OSError):
            raise HTTPException(400, "could not decode this audio file")
        del raw
    else:
        raise HTTPException(400, "bad source")
    if not BUSY.acquire(blocking=False):
        raise HTTPException(429, "another call is being scored; try again in a few seconds")

    q = queue.Queue()

    def work():
        try:
            a8 = to_8k(audio16[: 60 * 16000], channel)
            t0 = time.time()
            n_steps = int(np.ceil(len(a8) / 8000 / 10))
            q.put({"type": "start", "n_steps": n_steps, "duration": len(a8) / 8000, "alert_thr": {k: v["thr5"] for k, (m, v) in sc.models.items()}})

            def on_step(t, text, cur):
                q.put({"type": "step", "t": t, "text": text, "scores": cur})

            res = sc.score(a8, on_step)
            el = time.time() - t0
            times, fused, text = res["times"], res["scores"]["fused"], res["scores"]["text"]
            vf, cf = verdict(fused[-1], res["final_thr"]["fused"]); vt, ct = verdict(text[-1], res["final_thr"]["text"])
            out = {"type": "result", "times": times, "scores": res["scores"], "alert_thr": res["alert_thr"], "elapsed": el, "duration": len(a8) / 8000,
                   "fused": {"verdict": vf, "confidence": cf, "first_alert": first_alert(times, fused, res["alert_thr"]["fused"])},
                   "text": {"verdict": vt, "confidence": ct, "first_alert": first_alert(times, text, res["alert_thr"]["text"])},
                   "transcript": res["transcripts"][-1] if res["transcripts"] else ""}
            if truth is not None:
                out["truth"] = {"label": int(truth["label"]), "note": truth["note"]}
            q.put(out)
        except Exception as e:
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(None); BUSY.release()

    threading.Thread(target=work, daemon=True).start()

    async def gen():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


HYPOTHESES = [
    {"id": "H1", "claim": "Phone-channel degradation and recognition errors reduce transcript-based detection", "verdict": "partly supported",
     "evidence": "Word error rate stays 14-16 % for codecs and packet loss (13.5 % clean) but reaches 34 % at 5 dB noise and 66 % in the hardest condition. Text AUROC falls 0.96 to 0.91 to 0.80."},
    {"id": "H2", "claim": "An acoustic branch recovers part of the loss", "verdict": "not confirmed",
     "evidence": "Audio-only AUROC is about 0.55. Fusion minus text-only over 5 seeds: mean over 13 degraded conditions F1 -0.003 (95 % CI -0.022 to +0.016); every interval includes zero."},
    {"id": "H3", "claim": "Channel augmentation makes models more robust", "verdict": "not confirmed",
     "evidence": "Over 5 seeds hard-condition AUROC is +0.03 (text) and +0.025 (fusion) with augmentation, but every confidence interval includes zero."},
    {"id": "H4", "claim": "The fused model flags fraud earlier at the same false-alarm rate", "verdict": "not confirmed",
     "evidence": "Share of fraud calls flagged by 40 s: fused 0.40 vs text 0.44 over 5 seeds (paired p = 0.65). The 3-seed edge (0.48 vs 0.40) was seed luck."},
    {"id": "H5", "claim": "Models trained on one TTS system hold up on an unseen one", "verdict": "partly supported",
     "evidence": "Text and fused F1 changed by about 0.02 on audio from a different TTS system; the clean-trained audio-only model fell from 0.67 to 0.23."},
]


@app.get("/api/results")
def results():
    S = pd.read_csv("results/summary_by_cond.csv"); S = S[S.set == 1]
    labels = {"T-clean": "Text, original transcript", "T-asr": "Text, recognised speech", "A": "Audio only", "F": "Fusion", "F-aug": "Fusion + augmentation"}
    rows = []
    for m, lab in labels.items():
        r = {"model": m, "label": lab}
        for c in ("clean", "snr5", "hard"):
            g = S[(S.model == m) & (S.cond == c)]
            r[c] = None if g.empty else {"f1": float(g.f1_mean.iloc[0]), "f1_std": float(g.f1_std.iloc[0]), "auroc": float(g.auroc_mean.iloc[0])}
        rows.append(r)
    W = pd.read_csv("results/wer_by_cond.csv"); W = W[W.set == 1]
    wer = [{"cond": r.cond, "wer": float(r["mean"])} for _, r in W.iterrows()]
    return {"main": rows, "wer": wer, "hypotheses": HYPOTHESES,
            "figures": [{"src": "/figures/h1_f1_vs_severity.png", "caption": "F1 across the phone-channel grid (clean-trained models)"},
                        {"src": "/figures/wer_by_condition.png", "caption": "Word error rate by channel condition"},
                        {"src": "/figures/h4_time_to_detection_clean.png", "caption": "Fraud calls flagged against time (clean audio)"},
                        {"src": "/figures/h5_unseen_tts.png", "caption": "Seen versus unseen text-to-speech system"}]}


app.mount("/figures", StaticFiles(directory="figures"), name="figures")
app.mount("/", StaticFiles(directory="web", html=True), name="web")
