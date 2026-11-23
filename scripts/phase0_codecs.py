"""Phase 0: check which telephony codecs this machine's ffmpeg can round-trip."""
import subprocess, json, os, time, soundfile as sf, numpy as np
src = "tmp_phase0/piper_en_libritts_r_en.wav"
os.makedirs("tmp_phase0/codec", exist_ok=True)
specs = {
 "g711_mulaw": (["-ar","8000","-ac","1","-c:a","pcm_mulaw"], "wav"),
 "g711_alaw":  (["-ar","8000","-ac","1","-c:a","pcm_alaw"], "wav"),
 "gsm":        (["-ar","8000","-ac","1","-c:a","libgsm"], "gsm"),
 "opus_8k":    (["-ar","8000","-ac","1","-c:a","libopus","-b:a","8k"], "ogg"),
 "opus_12k":   (["-ar","8000","-ac","1","-c:a","libopus","-b:a","12k"], "ogg"),
 "amr_nb":     (["-ar","8000","-ac","1","-c:a","libopencore_amrnb","-b:a","12.2k"], "amr"),
}
res = {}
for k,(args,ext) in specs.items():
    enc = f"tmp_phase0/codec/{k}.{ext}"; dec = f"tmp_phase0/codec/{k}_dec.wav"
    t=time.time()
    r1 = subprocess.run(["ffmpeg","-y","-loglevel","error","-i",src,*args,enc],capture_output=True,text=True)
    r2 = subprocess.run(["ffmpeg","-y","-loglevel","error","-i",enc,"-ar","8000","-ac","1","-c:a","pcm_s16le",dec],capture_output=True,text=True)
    ok = r1.returncode==0 and r2.returncode==0 and os.path.exists(dec)
    d = {"ok": ok, "sec": round(time.time()-t,2)}
    if ok:
        a,sr = sf.read(dec); d.update(dur=round(len(a)/sr,2), rms=float(np.sqrt((a**2).mean())), bytes=os.path.getsize(enc))
    else: d["err"]=(r1.stderr+r2.stderr)[:200]
    res[k]=d; print(k,d)
json.dump(res, open("results/phase0_codecs.json","w"), indent=1)
