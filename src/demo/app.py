"""Streamlit demo: score a phone call in cumulative 10-second steps.   streamlit run src/demo/app.py

Input: an uploaded WAV/MP3, or one of 6 held-out test calls (label hidden until after scoring), with an
optional phone-channel degradation. Uploaded audio is processed in memory only and never stored.
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import altair as alt
from scipy.signal import resample_poly
from src.demo.scoring import Scorer, decode_to_16k, first_alert
from src.channel.channel import apply_channel, TEST_GRID
from src.demo import style

st.set_page_config(page_title="Fraud-call detector", page_icon="📞", layout="wide")
CHANNELS = {"clean (no degradation)": "clean", "G.711 mu-law": "mulaw", "GSM": "gsm", "Opus 8 kbps": "opus8",
            "noisy (SNR 10 dB)": "snr10", "lossy (10 % packets lost)": "loss10"}
NAMES = {"text": "Text only (BiLSTM)", "audio": "Audio only (CNN-BiGRU)", "fused": "Fused"}


@st.cache_resource(show_spinner="Loading models...")
def get_scorer():
    return Scorer()


def to_8k(audio16k, channel_key):
    if channel_key == "clean":
        return resample_poly(audio16k, 1, 2).astype(np.float32)
    return apply_channel(audio16k, TEST_GRID[channel_key], seed=0)


def verdict(score, thr):
    flag = score >= thr
    return ("FRAUD LIKELY" if flag else "No fraud detected"), (score if flag else 1 - score)


style.apply()
sc = get_scorer(); dc = sc.dc
samples = json.load(open("results/demo_samples.json", encoding="utf8"))

_, mid, _ = st.columns([1, 3, 1])
with mid:
    src = st.radio("Source", ["Held-out test call", "Upload WAV / MP3"], horizontal=True)
    audio16, label_hidden, truth = None, None, None
    if src == "Held-out test call":
        pick = st.selectbox("Sample", [s["id"] for s in samples])
        s = next(x for x in samples if x["id"] == pick)
        audio16, _ = sf.read(s["wav"], dtype="float32"); truth = s
        st.audio(s["wav"])
    else:
        up = st.file_uploader("Audio file (first 60 s are used)", type=["wav", "mp3"])
        if up is not None:
            audio16 = decode_to_16k(up.getvalue(), os.path.splitext(up.name)[1] or ".wav")   # in memory / temp dir deleted at once
    ch = st.selectbox("Phone channel applied before scoring", list(CHANNELS))
    go = st.button("Score call", type="primary", disabled=audio16 is None)

if go and audio16 is not None:
    a8 = to_8k(audio16[: 60 * 16000], CHANNELS[ch])
    t0 = time.time()
    left, right = st.columns([1, 1])
    prog = st.progress(0.0, text="Recognising speech ...")
    ph_tx, ph_chart = left.empty(), right.empty()
    n_steps = int(np.ceil(len(a8) / 8000 / 10)); live = {"t": [], "text": "", "s": {k: [] for k in sc.models}}

    def draw(df):
        long = df.melt("t", var_name="model", value_name="score")
        line = alt.Chart(long).mark_line(point=True).encode(x=alt.X("t:Q", title="seconds into call"), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="fraud score"),
                                                           color=alt.Color("model:N", title=None))
        rule = alt.Chart(pd.DataFrame({"y": [dc["models"]["fused"]["thr5"]]})).mark_rule(strokeDash=[5, 4], color="#ff5d73").encode(y="y:Q")
        pal = alt.Scale(domain=[NAMES["fused"], NAMES["text"], NAMES["audio"]], range=["#a99cff", "#35e0c2", "#ffb454"])
        line = line.encode(color=alt.Color("model:N", title=None, scale=pal))
        return (line + rule).configure(background="transparent").configure_axis(gridColor="#24243a", labelColor="#c4c4de", titleColor="#c4c4de", domainColor="#34345a").configure_legend(labelColor="#e6e6f5")

    def on_step(t, text, cur):
        live["t"].append(t); live["text"] = text
        for k, v in cur.items():
            live["s"][k].append(v)
        ph_tx.markdown(style.transcript_html(text, t, live=True), unsafe_allow_html=True)
        df = pd.DataFrame({"t": live["t"], **{NAMES[k]: v for k, v in live["s"].items()}})
        ph_chart.altair_chart(draw(df), width="stretch")
        prog.progress(min(1.0, len(live["t"]) / n_steps), text=f"Scored {t:.0f} s of the call")

    res = sc.score(a8, on_step)
    el = time.time() - t0
    prog.empty()
    ph_tx.markdown(style.transcript_html(live.get("text"), live["t"][-1], live=False), unsafe_allow_html=True)
    times = res["times"]; fused = res["scores"]["fused"]; text = res["scores"]["text"]; audio = res["scores"]["audio"]
    vf, cf = verdict(fused[-1], res["final_thr"]["fused"]); vt, ct = verdict(text[-1], res["final_thr"]["text"])
    af = first_alert(times, fused, res["alert_thr"]["fused"]); at = first_alert(times, text, res["alert_thr"]["text"])
    st.toast("Scoring finished", icon="✅")
    tab1, tab2, tab3 = st.tabs(["Verdict", "Score timeline", "Text vs fused"])
    with tab1:
        st.markdown(style.verdict_html(vf, cf) + style.stats_html(af, el, len(a8) / 8000, vt == vf), unsafe_allow_html=True)
        if truth is not None:
            st.markdown(f'<div class="truth">Ground truth for this held-out call: <b>{"FRAUD" if truth["label"] == 1 else "normal"}</b> &nbsp;({truth["note"]})</div>',
                        unsafe_allow_html=True)
    with tab2:
        st.markdown("##### Each 10-second step: red = fused score above the alert threshold")
        st.markdown(style.timeline_html(times, fused, res["alert_thr"]["fused"]), unsafe_allow_html=True)
        st.markdown("##### Final scores by model")
        st.markdown(style.meters_html([("Fused", "fusedc", fused[-1], res["alert_thr"]["fused"]),
                                       ("Text only (BiLSTM)", "textc", text[-1], res["alert_thr"]["text"]),
                                       ("Audio only (CNN-BiGRU)", "audioc", audio[-1], res["alert_thr"]["audio"])]), unsafe_allow_html=True)
    with tab3:
        cmp = pd.DataFrame({"model": ["Text only", "Fused"], "final verdict": [vt, vf], "confidence": [f"{ct:.0%}", f"{cf:.0%}"],
                            "first alert": [f"{at:.0f} s" if at is not None else "never", f"{af:.0f} s" if af is not None else "never"]})
        st.table(cmp)
        st.caption("Dashed line in the chart above: alert threshold (the score at which only 5 % of normal validation calls ever fire). "
                   "Highlighted words in the transcript come from a simple keyword list and are not what the model used.")

st.markdown('<div class="foot">This system was trained and evaluated on synthetic phone audio generated from dataset transcripts '
            '(text-to-speech plus a simulated phone line). It may behave very differently on real calls. Research prototype: '
            'not a security product. Uploaded audio is processed in memory and never stored.</div>', unsafe_allow_html=True)
