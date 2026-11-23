# Robust Fraud-Call Detection on Indian Phone Speech

A research project that measures how much phone-line degradation and speech-recognition errors hurt transcript-based fraud-call detection, and whether an acoustic branch and early-stage detection recover the loss.

> **Limitation.** The system is trained and evaluated on synthetic speech generated from dataset transcripts, so results may not transfer to real calls. Not a security product.

## What it does
1. Takes 800 balanced conversations from ICFD-31k (Roman-script English/Hinglish), split by `session_id`.
2. Renders each conversation as a call with a text-to-speech model (Piper; MMS-TTS for an unseen test set), one voice per speaker.
3. Degrades the audio with a simulated phone channel (8 kHz, G.711, GSM, AMR-NB, Opus, noise, packet loss) and transcribes it with faster-whisper.
4. Trains a BiLSTM text model, a CNN-BiGRU audio model and their fusion, evaluates them on a 14-condition grid, time-to-detection at 5 % false alarms and an unseen-TTS set, 3 seeds each.
5. Serves a local Streamlit demo that scores a call in cumulative 10-second steps.

Plain-language notes on every component: `docs/explain/`. Phase 0 findings (hardware, data, TTS, codecs, ASR): `docs/phase0_report.md`. Licences: `docs/licenses.md`. Viva answers: `docs/viva.md`.

## Results summary (synthetic audio, 124 test calls, 3 seeds, mean)
| model | clean F1 / AUROC | SNR 5 dB | hard (Opus 8k + 5 dB + 10 % loss) |
|---|---|---|---|
| T-clean (original text) | 0.858 / 0.963 | same | same |
| T-asr (recognised text) | 0.881 / 0.964 | 0.845 / 0.914 | 0.676 / 0.795 |
| A (audio only) | 0.668 / 0.572 | 0.000 / 0.534 | 0.199 / 0.552 |
| F (fusion) | 0.874 / 0.959 | 0.837 / 0.915 | 0.703 / 0.795 |
| F-aug (fusion + channel augmentation) | 0.857 / 0.955 | 0.803 / 0.914 | 0.632 / 0.805 |

Codecs and packet loss barely hurt; noise-driven recognition errors do. The audio branch was near chance and fusion matched text-only (five-seed paired tests: difference within about 0.02 AUROC, every confidence interval includes zero): H2, H3, H4 are not supported in this synthetic setting. Details, confidence intervals, ablations and error analysis: `report/report.md`, `results/tables.md`.

## Setup
Python 3.11, CPU only (no GPU needed), ffmpeg on PATH.
```bash
pip install -r requirements.txt
```
Download the Piper voices once into `data/models/piper/` (`en_US-libritts_r-medium` from `rhasspy/piper-voices` on Hugging Face) and the ICFD-31k `source_conversations/` into `data/raw/icfd/` (see the dataset's licence terms in `docs/licenses.md`).

## One command per stage
```bash
make build-data   # sample 800 conversations, synthesize, apply channels, ASR, manifest, features, statistics (~4-5 h on 10 CPU cores)
make train        # all experiments x 3 seeds, then evaluation (~7 h on 10 CPU cores, run in the background)
make eval         # metric tables and figures (src/eval/run_eval.py, src/eval/figures.py)
make demo         # streamlit run src/demo/app.py
make test         # unit tests (no session overlap, deterministic channel, model shapes)
```
Extra scripts: `python -m scripts.make_demo` (checkpoints, thresholds, 6 samples), `python -m scripts.demo_backup` (records demo output on the 6 samples), `python -m scripts.demo_headless_test` (runs the Streamlit app headlessly), `bash scripts/train_extra_seeds.sh` then `python -m src.eval.extra_analysis` (seeds 3-4 and paired tests / alert rule), `python -m src.eval.calibration_threshold`, `python -m src.eval.tok512_compare`, `bash scripts/train_extra_seeds_aug.sh` (augmented-model seeds 3-4), `python -m src.eval.error_analysis`, `python -m scripts.make_slides`, `python -m scripts.md_to_docx` (report.docx; report.pdf via LibreOffice).

## Layout
`configs/` YAML per experiment; `src/data` ICFD loading, splits, manifest builder; `src/tts` TTS wrappers and call assembly; `src/channel` phone-channel simulation; `src/asr` recogniser and WER; `src/features` log-mel and tokenization; `src/models` branches and fusion; `src/train` training loop; `src/eval` metrics, figures, error analysis; `src/demo` Streamlit app; `results/` logged runs and tables; `figures/` plots; `report/` report and slides; `notebooks/` Colab notebook for the optional T-rob baseline (not run).

## Reproducibility and privacy
Fixed seeds, YAML configs, pinned `requirements.txt`. Data, generated audio and checkpoints are git-ignored and not published. The demo never stores uploaded audio.
