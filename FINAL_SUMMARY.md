# Final summary

**Limitation (required).** The system is trained and evaluated on synthetic speech generated from dataset transcripts, so results may not transfer to real calls.

## What was built
A full pipeline for robust fraud-call detection on phone speech, CPU only:
* **Data build:** 800 balanced ICFD-31k conversations (402 sessions, 30 scenarios) split by `session_id` (557 / 119 / 124); Piper TTS calls (24-voice pool, caller pace from dataset labels), a deterministic phone-channel simulator (8 kHz, G.711 mu-law/A-law, GSM, AMR-NB, Opus 8/12 kbps, noise, packet loss), faster-whisper base ASR, manifest of 4,260 call variants (4.0 GB audio), plus an unseen-TTS test set (MMS-TTS, 124 calls x 4 conditions).
* **Models:** BiLSTM text branch (1.51 M params), CNN-BiGRU audio branch (0.11 M), early fusion (1.68 M), late fusion, with and without channel augmentation, 3 seeds each.
* **Evaluation:** 14-condition channel grid, F1/AUROC/PR-AUC/recall at 5 % false alarms, calibration, session-cluster bootstrap CIs, time-to-detection, unseen-TTS test, ablations, error analysis (251 errors grouped by cause, 5 hand-checked cases).
* **Demo:** local Streamlit app (upload or 6 held-out samples, channel selector, live transcript and score chart, text-only vs fused comparison); verified headless and with a backup recording script.
* **Documents:** `report/report.md` (+ `.docx`, `.pdf`), `report/slides.pptx` (15 slides), `docs/viva.md`, `docs/resume_lines.md`, `docs/explain/`, `docs/licenses.md`, `docs/phase0_report.md`, `PROGRESS.md`.

## Key results (test split, 124 calls, 3 seeds, mean ± sd; fraud positive)
| model | clean F1 / AUROC | SNR 5 dB F1 / AUROC | hard F1 / AUROC |
|---|---|---|---|
| T-clean | 0.858±0.045 / 0.963 | same | same |
| T-asr | 0.881±0.028 / 0.964 | 0.845±0.037 / 0.914 | 0.676±0.061 / 0.795 |
| A | 0.668±0.039 / 0.572 | 0.000 / 0.534 | 0.199±0.345 / 0.552 |
| F | 0.874±0.008 / 0.959 | 0.837±0.038 / 0.915 | 0.703±0.052 / 0.795 |
| F-aug | 0.857±0.034 / 0.955 | 0.803±0.042 / 0.914 | 0.632±0.125 / 0.805 |

| hypothesis | verdict | evidence |
|---|---|---|
| H1 channel + ASR hurt text detection | partly supported | WER 14-16 % for codecs/loss vs 13.5 % clean, 34 % at 5 dB, 66 % hard; AUROC 0.96 -> 0.91 (5 dB) -> 0.80 (hard); codecs alone cost nothing measurable |
| H2 acoustic branch recovers loss | not supported | audio-only AUROC ~0.55; F - T-asr over 5 seeds: mean of 13 degraded F1 -0.003 (95 % CI -0.022 to +0.016), AUROC -0.002 (-0.017 to +0.012), hard F1 +0.017 (-0.048 to +0.083); no paired p < 0.10 |
| H3 augmentation more robust | not supported | 3 seeds: F-aug hard F1 0.632 vs F 0.703 (within noise); 5 seeds: hard AUROC +0.031 (text) / +0.025 (fusion) for augmentation, all CIs include 0, so no significant gain |
| H4 fused flags earlier | not supported | flagged by 40 s: 3 seeds 0.476 vs 0.402, but 5 seeds 0.400 vs 0.435 (paired p = 0.65); a k-consecutive-step alert rule trades earliness for recall and does not favour fusion |
| H5 unseen TTS | partly supported | clean F1 T-asr 0.881 -> 0.859, F 0.874 -> 0.869; clean-trained audio model 0.668 -> 0.231 |

## What could not be done, and why
* **T-rob transformer baseline: not run.** Estimated CPU cost too high; Colab notebook `notebooks/T-rob_colab.ipynb` written but not executed (run it on a GPU runtime, upload `data/manifest/manifest.jsonl`, copy `results/runs/T-rob/` back, then `python -m src.eval.run_eval`).
* **Telugu and Devanagari Hindi:** dropped. ICFD is Roman-script; Whisper tiny/base cannot transcribe Hindi (WER > 1); no Telugu TTS/ASR was tested.
* **Genuine acoustic signal:** the injected pace cue is weak (measured speaking-rate AUROC 0.52), so the audio branch result says little about real acoustic cues.
* **Extra seeds and alert rule (added):** T-asr, A and F have 5 seeds (`scripts/train_extra_seeds.sh`); `python -m src.eval.extra_analysis` writes `results/paired_tests.csv`, `results/alert_rule_summary.csv`, `results/extra_analysis.md`. Primary tables stay at seeds 0-2.
* **More extras (added):** `python -m src.eval.calibration_threshold` (temperature scaling and threshold drift, `results/calibration_threshold.md`), `python -m src.eval.tok512_compare` (512 vs 256 text tokens, `results/tok512_compare.md`), `bash scripts/train_extra_seeds_aug.sh` (seeds 3-4 for the augmented models, H3 paired tests in `results/extra_analysis.md`).
* **Ablations not run:** each channel factor trained alone (cut by the brief's priority order) (it needs new degraded training audio and about 2.4 h of recognition); the 512-token text check was run later and shows no gain (`results/tok512_compare.md`). Run: audio/text branch removal, late vs early fusion, 10/30/60 s audio window, 128 vs 256 tokens, augmentation on/off (report Section 6).
* **Official ICFD splits** could not be used (99.5 % fraud in validation/test); we re-split. cross_domain was not used (it shares `session_id`s with train and validation).
* **Real-call validation:** none; all audio is synthetic.
* **Demo screenshots:** the slides use a figure of the demo pipeline's recorded output on the 6 samples, not browser screenshots; the app itself was run headlessly (`results/demo_headless.json`).

## Reproduce each stage
```bash
pip install -r requirements.txt
python -m src.data.icfd                                   # balanced sample + session split -> data/processed/conversations.json
make build-data                                           # synth (Piper), channel + ASR, Set 2 (MMS), manifest, features, stats
python -m pytest tests -q                                 # 15 tests
make train                                                # 11 experiments x 3 seeds + evaluation
python -m src.eval.run_eval && python -m src.eval.figures # metric tables and figures -> results/, figures/
python -m src.eval.error_analysis                         # error analysis
python -m scripts.make_demo && make demo                  # demo config + 6 samples, then streamlit
python -m scripts.demo_backup                             # backup recording of demo output
python -m scripts.make_slides && python -m scripts.md_to_docx    # slides, report docx/pdf
```
Measured cost on this machine (10 cores, 15.7 GB RAM, no GPU): data build about 4 h (synthesis 22 min, ASR 2.4 h, MMS 48 min), training about 7 h, demo scoring 3.2-4.5 s for a 52-60 s call on an idle CPU, 12-38 s while training jobs were running (`results/demo_backup.json`). Demo scores can vary slightly between runs (int8 ASR depends on thread scheduling).
