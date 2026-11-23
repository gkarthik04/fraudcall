"""Visual style and small HTML components for the demo (dark page, indigo/teal accents, animations)."""
import html
import re
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
:root { --bg:#0a0a14; --panel:#15152a; --panel2:#1b1b36; --line:#34345a; --accent:#7c6cff; --accent2:#a99cff;
        --teal:#35e0c2; --amber:#ffb454; --red:#ff5d73; --text:#f3f3fb; --muted:#b4b4d0; }
html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, label, li { font-family:'Inter',sans-serif !important; color:var(--text); }
[data-testid="stAppViewContainer"] { background: var(--bg); overflow:hidden; position:relative; }
/* animated glow orbs */
[data-testid="stAppViewContainer"]::before, [data-testid="stAppViewContainer"]::after {
  content:""; position:fixed; width:620px; height:620px; border-radius:50%; filter:blur(90px); opacity:.38; z-index:0; pointer-events:none; }
[data-testid="stAppViewContainer"]::before { background:var(--accent); left:-180px; top:5%; animation:drift1 16s ease-in-out infinite alternate; }
[data-testid="stAppViewContainer"]::after  { background:#1fc7b0; right:-200px; top:30%; opacity:.22; animation:drift2 19s ease-in-out infinite alternate; }
@keyframes drift1 { to { transform:translate(120px,90px) scale(1.15); } }
@keyframes drift2 { to { transform:translate(-140px,-60px) scale(1.2); } }
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none; }
.block-container { max-width:1080px; padding-top:1rem; position:relative; z-index:1; }
/* undo Streamlit's dimming of elements while the script reruns */
[data-stale="true"], [data-testid="stElementContainer"][data-stale="true"] { opacity:1 !important; transition:none !important; }

h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em; color:var(--text) !important; }
.topbar { display:flex; align-items:center; gap:26px; padding:10px 6px 14px; border-bottom:1px solid var(--line); font-size:.92rem;
          animation:fadeDown .6s ease both; }
.topbar a, .topbar span.nav { color:var(--muted); text-decoration:none; position:relative; cursor:pointer; transition:color .2s; }
.topbar .nav:hover { color:#fff; }
.topbar .nav::after { content:""; position:absolute; left:0; bottom:-6px; height:2px; width:0; background:var(--accent2); transition:width .25s; }
.topbar .nav:hover::after { width:100%; }
.brand { font-family:'Space Grotesk',sans-serif; color:#fff; font-weight:700; font-size:1.15rem; margin-right:10px; }
.brand b { color:var(--teal); }
.topbar .right { margin-left:auto; display:flex; gap:14px; align-items:center; color:var(--muted); }
.pill { background:linear-gradient(135deg,var(--accent),#5b4cf0); color:#fff; padding:6px 14px; border-radius:999px; font-weight:600; font-size:.82rem;
        box-shadow:0 0 0 0 rgba(124,108,255,.6); animation:ring 2.4s infinite; }
@keyframes ring { 70% { box-shadow:0 0 0 12px rgba(124,108,255,0); } 100% { box-shadow:0 0 0 0 rgba(124,108,255,0); } }
@keyframes fadeDown { from { opacity:0; transform:translateY(-12px); } }
@keyframes fadeUp { from { opacity:0; transform:translateY(18px); } }
.hero { text-align:center; padding:54px 0 18px; }
.hero > * { animation:fadeUp .8s cubic-bezier(.2,.7,.2,1) both; }
.hero .tag { display:inline-block; border:1px solid var(--line); background:rgba(124,108,255,.12); color:var(--accent2); padding:5px 14px;
             border-radius:999px; font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; margin-bottom:20px; animation-delay:.05s; }
.hero h1 { font-size:clamp(2.4rem,6vw,4.4rem); line-height:1.04; font-weight:700; margin:0 0 18px; padding:0; animation-delay:.15s;
           background:linear-gradient(100deg,#fff 20%,#b9aeff 45%,#35e0c2 60%,#fff 80%); background-size:240% 100%;
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:fadeUp .8s .15s both, shimmer 7s linear infinite; }
@keyframes shimmer { to { background-position:-240% 0; } }
.hero p { color:var(--muted); font-size:1.12rem; max-width:580px; margin:0 auto; animation-delay:.3s; }
.steps { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:34px 0 8px; }
.step { background:rgba(21,21,42,.8); border:1px solid var(--line); border-radius:14px; padding:14px 14px 12px; transition:transform .25s, border-color .25s, box-shadow .25s;
        animation:fadeUp .7s both; }
.step:nth-child(1){animation-delay:.4s}.step:nth-child(2){animation-delay:.5s}.step:nth-child(3){animation-delay:.6s}.step:nth-child(4){animation-delay:.7s}
.step:hover { transform:translateY(-5px); border-color:var(--accent); box-shadow:0 12px 30px -10px rgba(124,108,255,.55); }
.step .n { font-family:'JetBrains Mono',monospace; color:var(--teal); font-size:.8rem; }
.step .t { font-family:'Space Grotesk',sans-serif; font-weight:600; margin:4px 0 2px; }
.step .d { color:var(--muted); font-size:.82rem; line-height:1.35; }
@media (max-width:760px){ .steps{grid-template-columns:1fr 1fr;} }

/* controls */
.stButton>button { border-radius:12px; background:linear-gradient(135deg,var(--accent),#5b4cf0); border:0; color:#fff; font-weight:600; font-family:'Space Grotesk',sans-serif;
                   padding:.65rem 1.6rem; width:100%; transition:transform .15s, box-shadow .2s; }
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 10px 26px -8px rgba(124,108,255,.8); color:#fff; }
.stButton>button:active { transform:scale(.98); }
div[data-baseweb="select"]>div, [data-testid="stFileUploaderDropzone"] { background:var(--panel2) !important; border:1px solid var(--line) !important; border-radius:12px !important; }
div[data-baseweb="select"]>div:hover { border-color:var(--accent) !important; }
[data-testid="stRadio"] label p, [data-testid="stWidgetLabel"] p { color:var(--muted) !important; font-weight:500; }
[data-testid="stTabs"] button p { font-family:'Space Grotesk',sans-serif; font-size:1rem; }
[data-testid="stTabs"] button[aria-selected="true"] p { color:#fff !important; }
[data-baseweb="tab-highlight"] { background:linear-gradient(90deg,var(--accent),var(--teal)) !important; height:3px !important; }

/* live transcript */
.tx { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:12px; padding:14px 16px; color:#e9e9f7;
      line-height:1.65; font-size:.97rem; max-height:360px; overflow:auto; animation:fadeUp .5s both; }
.tx .cap { font-family:'JetBrains Mono',monospace; color:var(--teal); font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; display:block; margin-bottom:6px; }
.tx mark { background:rgba(255,93,115,.22); color:#ffd2d9; border-radius:5px; padding:0 4px; border-bottom:2px solid var(--red); transition:background .2s; }
.tx mark:hover { background:rgba(255,93,115,.5); }
.caret::after { content:"|"; color:var(--accent2); animation:blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity:0; } }

/* verdict card */
.verdict { display:flex; align-items:center; gap:20px; border-radius:18px; padding:20px 22px; border:1px solid var(--line); background:var(--panel);
           animation:pop .6s cubic-bezier(.2,1.2,.3,1) both; position:relative; overflow:hidden; }
.verdict.fraud { border-color:rgba(255,93,115,.7); box-shadow:0 0 40px -8px rgba(255,93,115,.5); background:linear-gradient(135deg,#2a1424,var(--panel)); }
.verdict.ok { border-color:rgba(53,224,194,.6); box-shadow:0 0 40px -10px rgba(53,224,194,.45); background:linear-gradient(135deg,#0f2a2a,var(--panel)); }
.verdict .k { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }
.verdict .v { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.6rem,4vw,2.4rem); font-weight:700; line-height:1.1; }
.verdict.fraud .v { color:#ff8fa0; } .verdict.ok .v { color:var(--teal); }
.ring { --p:0; width:92px; height:92px; border-radius:50%; flex:none; display:grid; place-items:center; animation:fill 1.4s .2s ease-out forwards;
        background:conic-gradient(var(--c) calc(var(--p)*1%), rgba(255,255,255,.08) 0); }
.fraud .ring { --c:var(--red); } .ok .ring { --c:var(--teal); }
.ring span { width:72px; height:72px; border-radius:50%; background:var(--panel); display:grid; place-items:center;
             font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.05rem; }
@keyframes fill { to { --p:var(--target); } }
@keyframes pop { from { opacity:0; transform:scale(.92); } }
.dot { width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:8px; animation:pulse 1.4s infinite; }
.fraud .dot { background:var(--red); } .ok .dot { background:var(--teal); }
@keyframes pulse { 0% { box-shadow:0 0 0 0 currentColor; opacity:1; } 70% { box-shadow:0 0 0 10px transparent; } 100% { opacity:.7; } }

/* stat cards */
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:12px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; transition:transform .2s, border-color .2s; animation:fadeUp .6s both; }
.stat:nth-child(2){animation-delay:.1s}.stat:nth-child(3){animation-delay:.2s}
.stat:hover { transform:translateY(-4px); border-color:var(--accent); }
.stat .k { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
.stat .v { font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:700; margin-top:4px; }
.stat .s { color:var(--muted); font-size:.8rem; }
@media (max-width:760px){ .stats{grid-template-columns:1fr;} }

/* meters */
.meter { margin:12px 0; animation:fadeUp .6s both; }
.meter .row { display:flex; justify-content:space-between; font-size:.9rem; margin-bottom:6px; }
.meter .row b { font-family:'JetBrains Mono',monospace; }
.bar { position:relative; height:14px; background:rgba(255,255,255,.08); border-radius:999px; overflow:hidden; }
.bar i { position:absolute; inset:0 auto 0 0; width:var(--w); border-radius:999px; animation:grow 1.2s cubic-bezier(.2,.8,.2,1) both; }
.bar i.fusedc { background:linear-gradient(90deg,#6d5cff,#b3a8ff); } .bar i.textc { background:linear-gradient(90deg,#1fb59f,#35e0c2); } .bar i.audioc { background:linear-gradient(90deg,#d9902d,#ffb454); }
.bar u { position:absolute; top:-2px; bottom:-2px; width:2px; background:#fff; left:var(--t); opacity:.85; }
@keyframes grow { from { width:0; } }

/* timeline */
.tl { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.tl div { width:62px; text-align:center; padding:10px 0 8px; border-radius:12px; border:1px solid var(--line); background:var(--panel); font-family:'JetBrains Mono',monospace;
          font-size:.8rem; animation:pop .5s both; transition:transform .2s; cursor:default; }
.tl div:hover { transform:translateY(-4px) scale(1.06); }
.tl div b { display:block; font-size:1rem; margin-top:2px; }
.tl .hot { border-color:rgba(255,93,115,.7); background:rgba(255,93,115,.14); color:#ffc2cb; }
.tl .cool { border-color:rgba(53,224,194,.4); color:#a8f3e6; }
.truth { border:1px dashed var(--accent); border-radius:12px; padding:12px 16px; color:#dcd8ff; background:rgba(124,108,255,.1); margin-top:14px; animation:fadeUp .6s both; }
.foot { color:var(--muted); font-size:.82rem; text-align:center; border-top:1px solid var(--line); margin-top:44px; padding-top:16px; }
[data-testid="stTable"] table, .stTable table { color:var(--text); } [data-testid="stTable"] th { color:var(--muted); }
</style>
"""

TOPBAR = """<div class="topbar"><span class="brand">&#9742; Fraud<b>Call</b></span>
<span class="nav">Demo</span><span class="nav">Models</span><span class="nav">Report</span><span class="nav">Limits</span>
<span class="right"><span>Research prototype</span><span class="pill">Local &middot; private</span></span></div>
<div class="hero"><div class="tag">CNN&middot;BiGRU &nbsp;+&nbsp; BiLSTM &nbsp;+&nbsp; fusion</div>
<h1>Robust fraud-call<br>detection on phone speech</h1>
<p>Score a call as it unfolds, 10 seconds at a time, from the recognised words and the sound of the line.</p></div>
<div class="steps">
<div class="step"><div class="n">01</div><div class="t">Phone line</div><div class="d">8 kHz, codec, noise or lost packets applied to the call.</div></div>
<div class="step"><div class="n">02</div><div class="t">Recognise</div><div class="d">faster-whisper turns the audio into words, step by step.</div></div>
<div class="step"><div class="n">03</div><div class="t">Score</div><div class="d">Text, audio and fused models each give a fraud score.</div></div>
<div class="step"><div class="n">04</div><div class="t">Alert</div><div class="d">A threshold chosen on validation data fires the alert.</div></div></div>"""

# simple keyword list for highlighting only: it is NOT what the model used
KEYWORDS = r"(otp|password|pin|cvv|bank|account|card|upi|refund|kyc|urgent|immediately|arrest|police|link|verify|transfer|scam|lottery|prize|blocked|suspend\w*|whatsapp)"


def apply():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(TOPBAR, unsafe_allow_html=True)


def transcript_html(text, t, live=False):
    safe = html.escape(text or "...")
    safe = re.sub(r"\b" + KEYWORDS + r"\b", r"<mark>\1</mark>", safe, flags=re.I)
    cls = "tx caret" if live else "tx"
    return f'<div class="{cls}"><span class="cap">Transcript &middot; first {t:.0f} s</span>{safe}</div>'


def verdict_html(label, conf):
    cls = "fraud" if label.startswith("FRAUD") else "ok"
    return (f'<div class="verdict {cls}"><div class="ring" style="--target:{conf*100:.0f}"><span>{conf:.0%}</span></div>'
            f'<div><div class="k"><span class="dot"></span>Fused verdict</div><div class="v">{label}</div>'
            f'<div class="k" style="margin-top:4px">confidence {conf:.0%}</div></div></div>')


def stats_html(first_alert, elapsed, dur, agree):
    fa = f"{first_alert:.0f} s" if first_alert is not None else "never"
    return (f'<div class="stats"><div class="stat"><div class="k">First alert (fused)</div><div class="v">{fa}</div><div class="s">into the call</div></div>'
            f'<div class="stat"><div class="k">Processing time</div><div class="v">{elapsed:.1f} s</div><div class="s">for {dur:.0f} s of audio</div></div>'
            f'<div class="stat"><div class="k">Text vs fused</div><div class="v">{"agree" if agree else "differ"}</div><div class="s">final verdicts</div></div></div>')


def meters_html(rows):
    """rows: (label, css, final score, alert threshold)."""
    out = []
    for i, (label, css, s, thr) in enumerate(rows):
        out.append(f'<div class="meter" style="animation-delay:{i*0.12:.2f}s"><div class="row"><span>{label}</span><b>{s:.2f}</b></div>'
                   f'<div class="bar"><i class="{css}" style="--w:{s*100:.0f}%"></i><u style="--t:{thr*100:.0f}%"></u></div></div>')
    return "".join(out) + '<div style="color:var(--muted);font-size:.8rem">White tick: alert threshold (only 5 % of normal validation calls ever cross it).</div>'


def timeline_html(times, scores, thr):
    cells = []
    for i, (t, s) in enumerate(zip(times, scores)):
        cls = "hot" if s >= thr else "cool"
        cells.append(f'<div class="{cls}" style="animation-delay:{i*0.07:.2f}s" title="score {s:.2f}">{t:.0f}s<b>{s:.2f}</b></div>')
    return '<div class="tl">' + "".join(cells) + "</div>"
