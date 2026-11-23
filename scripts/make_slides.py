"""Build report/slides.pptx (python-pptx) from figures/ and results/. python -m scripts.make_slides"""
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

S = pd.read_csv("results/summary_by_cond.csv")
P = Presentation(); P.slide_width, P.slide_height = Inches(13.33), Inches(7.5)
BLANK = P.slide_layouts[6]; DARK = RGBColor(0x1F, 0x2A, 0x44); ACC = RGBColor(0x1B, 0x9E, 0x77)


def ms(m, c, k, st=1):
    r = S[(S.model == m) & (S.set == st) & (S.cond == c)]
    return float(r[k + "_mean"].iloc[0]), float(r[k + "_std"].iloc[0])


def slide(title, sub=None):
    s = P.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9)); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = DARK
    if sub:
        p2 = tf.add_paragraph(); p2.text = sub; p2.font.size = Pt(16); p2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return s


def bullets(s, items, x=0.6, y=1.6, w=12, h=5, size=20):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(size); p.space_after = Pt(8)


def img(s, path, x, y, w=None, h=None):
    s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def box(s, x, y, w, h, text, fill=ACC, size=14):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background()
    tf = b.text_frame; tf.word_wrap = True; tf.text = text
    for p in tf.paragraphs:
        p.font.size = Pt(size); p.font.color.rgb = RGBColor(255, 255, 255); p.alignment = PP_ALIGN.CENTER
    return b


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)); c.line.width = Pt(2); c.line.color.rgb = DARK


def table(s, rows, x, y, w, colw=None, size=13):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.4 * len(rows))).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size); p.font.bold = (i == 0)
    if colw:
        for j, cw in enumerate(colw):
            t.columns[j].width = Inches(cw)


# 1 title
s = slide("Robust Fraud-Call Detection on Indian Phone Speech", "How much do phone lines and speech recognition hurt transcript-based fraud detection, and does an acoustic branch help?")
bullets(s, ["Course project: deep learning (CNN/RNN backbones, generative AI for call audio)", "All audio is synthetic, generated from ICFD-31k transcripts: results may not transfer to real calls"], y=3.2)
# 2 problem
s = slide("The problem")
bullets(s, ["Published Indian fraud-call detectors (ICFD-31k benchmark) read clean text transcripts",
            "A real detector hears a phone call: 8 kHz audio, codec compression, noise, lost packets",
            "Speech recognition (ASR) then gives imperfect text, and the detector may miss fraud",
            "Question: how big is that loss, and can sound (pace, tone) recover it? And can we flag fraud earlier in the call?"])
# 3 why phone audio breaks
s = slide("Why phone audio breaks detectors", "Recognition word error rate by simulated channel condition (124 test calls)")
img(s, "figures/wer_by_condition.png", 1.2, 1.7, w=10.8)
bullets(s, ["Codecs and packet loss add ~1-2 points; noise at 5 dB reaches 34 %; the combined worst case reaches 66 %"], y=6.5, h=0.8, size=16)
# 4 RQ
s = slide("Research question and hypotheses")
bullets(s, ["H1  Detection accuracy falls as the channel worsens, and falls more after ASR than on clean transcripts",
            "H2  An acoustic branch recovers part of that loss",
            "H3  Training with channel augmentation is more robust than clean-only training",
            "H4  The fused model flags fraud earlier at the same false-alarm rate",
            "H5  A model trained on one TTS system still works on a different TTS system",
            "Negative results accepted if measured honestly"], size=20)
# 5 architecture
s = slide("Architecture", "Offline data build (top) and live scoring of one call (bottom)")
xs = [0.4, 2.8, 5.2, 7.6, 10.0]
labels = ["ICFD-31k transcript (turns, verdict)", "TTS: caller voice + callee voice (Piper)", "Phone channel: 8 kHz, codec, noise, packet loss", "ASR: faster-whisper base, words + times", "Cumulative 10 s steps + manifest"]
for x, t in zip(xs, labels):
    box(s, x, 1.7, 2.2, 1.2, t, size=13)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(s, a + 2.2, 2.3, b, 2.3)
box(s, 0.4, 4.2, 2.2, 1.0, "Call audio (WAV/MP3) -> 8 kHz mono", DARK, 13); box(s, 3.0, 3.8, 2.6, 0.9, "ASR -> transcript so far -> BiLSTM text branch", fill=RGBColor(0x1F, 0x77, 0xB4), size=13)
box(s, 3.0, 4.9, 2.6, 0.9, "Log-mel of last 30 s -> CNN-BiGRU audio branch", fill=RGBColor(0xD9, 0x5F, 0x02), size=13)
box(s, 6.2, 4.3, 2.4, 1.1, "Fusion MLP -> fraud score per step", fill=RGBColor(0x75, 0x70, 0xB3), size=13); box(s, 9.2, 4.3, 3.6, 1.1, "Score over time, first alert, final verdict (Streamlit demo)", DARK, 13)
arrow(s, 2.6, 4.7, 3.0, 4.25); arrow(s, 2.6, 4.7, 3.0, 5.3); arrow(s, 5.6, 4.25, 6.2, 4.7); arrow(s, 5.6, 5.3, 6.2, 5.0); arrow(s, 8.6, 4.85, 9.2, 4.85)
# 6 tech stack
s = slide("Tech stack and why")
table(s, [["Tool", "Role", "Why this one"],
          ["PyTorch (CPU)", "Models, training", "BiLSTM / CNN-BiGRU are small enough to train on a laptop"],
          ["Piper (VITS)", "TTS, 904 built-in voices", "3-14x real time on CPU; MMS-TTS is 0.8x"],
          ["MMS-TTS", "TTS for the unseen test set", "Different model family from Piper"],
          ["ffmpeg", "G.711, GSM, AMR-NB, Opus", "Real telephony codecs, deterministic"],
          ["faster-whisper base (int8)", "Speech recognition", "Runs ~7x real time on CPU; WER measured per condition"],
          ["XLM-R tokenizer", "Multilingual subwords", "Handles English + Hinglish spellings"],
          ["scikit-learn, jiwer", "Metrics, WER", "Standard, reproducible"],
          ["Streamlit", "Local demo", "One command, CPU only"]], 0.5, 1.5, 12.3, colw=[3.0, 3.8, 5.5], size=14)
# 7 data and channel
s = slide("Data and channel simulation")
bullets(s, ["800 ICFD-31k conversations, balanced fraud/normal, 402 sessions, 30 scenarios, 10 domains",
            "Split by session_id: train 557 / val 119 / test 124 (official val/test are 99.5 % fraud, so we re-split)",
            "Synthetic calls up to 60 s; caller pace follows the dataset's pace label (an injected, weak cue)",
            "Test grid of 14 conditions: clean, 8 kHz, mu-law, A-law, AMR-NB, GSM, Opus 12k/8k, loss 5/10 %, SNR 20/10/5 dB, combined worst case",
            "Training: clean only, or clean + 2 random channels (augmentation)",
            "Unseen TTS (MMS-TTS) rendering of the test calls in 4 conditions"], size=19)
# 8 models
s = slide("Models compared")
table(s, [["Model", "Input", "Trained on", "Params"],
          ["T-clean", "original clean transcript (BiLSTM)", "clean", "1.51 M"],
          ["T-asr", "recognised transcript (BiLSTM)", "clean audio", "1.51 M"],
          ["A", "log-mel, last 30 s (CNN-BiGRU)", "clean audio", "0.11 M"],
          ["F", "T-asr + A, concat + MLP", "clean audio", "1.68 M"],
          ["F-aug (and T-asr-aug, A-aug)", "same, with channel augmentation", "clean + 2 random channels", "1.68 M"],
          ["T-rob (optional)", "distilled multilingual transformer", "not run (Colab notebook provided)", "-"]], 0.5, 1.6, 12.3, colw=[3.2, 4.6, 3.2, 1.3], size=15)
bullets(s, ["3 seeds each, same splits, early stopping on validation F1, thresholds chosen on validation only"], y=5.2, h=1, size=16)
# 9 main table
s = slide("Main results (3 seeds, mean ± sd, test split)", "Call-level F1 / AUROC")
rows = [["Model", "clean", "SNR 5 dB", "hard (Opus8+5 dB+10 % loss)"]]
for m in ("T-clean", "T-asr", "A", "F", "T-asr-aug", "F-aug"):
    r = [m]
    for c in ("clean", "snr5", "hard"):
        f, fs = ms(m, c, "f1"); a, _ = ms(m, c, "auroc"); r.append(f"{f:.2f}±{fs:.2f} / {a:.2f}")
    rows.append(r)
table(s, rows, 0.6, 1.8, 12.1, colw=[2.5, 3.0, 3.0, 3.6], size=16)
bullets(s, ["Text carries the signal; the audio-only branch is near chance (AUROC ~0.55); fusion = text-only (5-seed paired test: all CIs include 0)"], y=5.6, h=1, size=18)
# 10 robustness plot
s = slide("Robustness across the channel grid (H1, H2)")
img(s, "figures/h1_f1_vs_severity.png", 1.0, 1.4, w=11.3)
# 11 TTD
s = slide("Time-to-detection at 5 % false alarms (H4)", "Share of fraud calls flagged against seconds into the call (clean audio)")
img(s, "figures/h4_time_to_detection_clean.png", 0.6, 1.6, h=4.9)
bullets(s, ["3 seeds: fused 0.48 vs text 0.40 by 40 s;", "5 seeds: 0.40 vs 0.44 (paired p = 0.65):", "the early gap was seed luck, H4 not supported", "Audio-only flags almost nothing"], x=7.6, y=2.0, w=5.3, size=18)
# 12 H3/H5
s = slide("Augmentation (H3) and unseen TTS (H5)")
img(s, "figures/h3_augmentation_hard.png", 0.4, 1.5, w=6.2); img(s, "figures/h5_unseen_tts.png", 6.7, 1.7, w=6.4)
bullets(s, ["H3: no significant gain from augmentation (5 seeds: hard AUROC +0.02-0.03, CIs include 0)", "H5: text/fused F1 within ~0.02 on a different TTS; clean-trained audio model collapsed (0.67 -> 0.23)"], y=5.6, h=1.5, size=16)
# 12b extra checks
s = slide("Extra checks with more seeds", "Paired by seed; 95 % confidence intervals (src/eval/extra_analysis.py)")
bullets(s, ["H2, 5 seeds: fusion minus text-only is about 0 (mean over 13 degraded: F1 -0.003, CI -0.022 to +0.016); every CI includes 0",
            "H4, 5 seeds: fused flags 0.40 vs text 0.44 of fraud calls by 40 s (p = 0.65); the 3-seed edge was seed luck",
            "H3, 5 seeds: augmentation hard AUROC +0.03 (text) / +0.025 (fusion), CIs include 0: weak hint, not significant",
            "512 vs 256 text tokens: no consistent gain (mean degraded AUROC 0.933 vs 0.944)",
            "Temperature scaling fitted on validation lowers ECE on noisy audio (hard: F 0.28 -> 0.19); a single threshold cannot fix drift"], size=18)
# 13 demo
s = slide("Demo (local Streamlit app)", "Upload WAV/MP3 or pick a held-out test call; choose a phone channel; see scores build up")
img(s, "figures/demo_output.png", 0.4, 1.9, w=7.8)
bullets(s, ["Output of the demo pipeline on the 6 prepared samples", "Runs on CPU; processing time measured in results/demo_headless.json", "Never stores uploaded audio"], x=8.4, y=2.0, w=4.6, size=16)
# 14 limitations
s = slide("Limitations")
bullets(s, ["Trained and evaluated on synthetic speech generated from dataset transcripts: results may not transfer to real calls",
            "Weak acoustic cue in synthetic voices (measured speaking-rate AUROC 0.52) explains the audio branch result",
            "Small ASR (Whisper base), 60 s cap, 30 scenarios, label noise in ICFD; 124 test calls, differences within noise not claimed",
            "T-rob transformer baseline, Telugu and Devanagari Hindi, and channel factors trained alone not done"], size=19)
# 15 conclusion
s = slide("Conclusion")
bullets(s, ["Codecs and packet loss barely hurt transcript detection; noise-driven recognition errors do (AUROC 0.96 -> 0.91 at 5 dB, 0.80 in the worst case)",
            "No evidence that the acoustic branch, augmentation or fusion recovers the loss in this synthetic setting (reported as negative results)",
            "Next: real recorded calls, genuine acoustic cues, stronger ASR, longer calls, Telugu/Hindi, streaming deployment"], size=20)
P.save("report/slides.pptx"); print(len(P.slides), "slides")
