# Progress log

## Phase 0: setup and day-one checks (done)
* Hardware: 10 cores, 15.7 GB RAM, 20 GB free, no GPU. Everything on CPU.
* ICFD-31k loaded. Official val/test splits are 99.5% fraud, and `session_id` covers 10 conversations each, so we pool and re-split by `session_id`. Licence permits this use (see docs/phase0_report.md).
* TTS: Piper libritts_r (3-14x real time) is Set 1; MMS-TTS-eng (0.8x) is Set 2. Hindi Devanagari ASR with Whisper tiny/base fails (WER > 1), data is Roman Hinglish anyway, so Telugu/Devanagari are out of scope.
* All six telephony codecs work in ffmpeg. ASR: faster-whisper base, WER 7-22% on clean synthetic English/Hinglish clips (3 clips only).
* Project folder moved to a short path (C:\Users\Karthik\fraudcall) because the scratch path exceeded the Windows 260-character limit.

## Phase 1: data pilot (done, gate passed)
* 150 balanced conversations (a stratified subset of the final 800), 703 call variants (clean + 2 random channels for train/val calls, 14-condition grid for test calls).
* Gate: 0 silent/short clips (all > 3 s), clean WER (whole call, faster-whisper base) 0.137 (limit 0.30), 12 tests pass, no session overlap.
* Cost: Piper synthesis 150 calls in 75 s with 5 workers (after capping ONNX threads at 2 per worker; the first attempt with uncapped threads was ~8x slower), ASR ~7.9 s per call-variant, 703 variants in 36 min with 3 workers. Clean audio 1.7 MB/call (16 kHz); whole pilot 0.72 GB.
* Early WER by condition (pilot, n=23 test calls each): codecs add only ~1 point over clean (0.137 vs 0.146-0.155), SNR 20/10/5 dB give 0.154/0.219/0.329, "hard" (Opus 8 kbps + 5 dB + 10 % loss) gives 0.672. Noise, not codec, drives recognition damage with this ASR.
* Note: 93 of 150 pilot calls are `normal` pace and only 1 is `slow`, so "pace" has mostly two levels in practice.

## Phase 2: full data build (done, gate passed)
* 800 conversations (402 sessions, 30 scenarios, all 10 domains), split by session_id: train 557 / val 119 / test 124 (balanced within each: 281/276, 58/61, 61/63 normal/fraud). 4,260 call variants: clean for all calls, 2 random channels per train/val call, the 14-condition grid for the 124 test calls, and TTS Set 2 (MMS-TTS, single voice with 5 pitch/rate variants) for the 124 test calls in 4 conditions.
* Gate: no silent/short clips, no session overlap (test passes on conversations and manifest), 15 tests pass.
* WER of ASR vs clean transcript (test, mean): clean 0.135; codecs 0.140-0.151; loss 5/10 % 0.148/0.156; SNR 20/10/5 dB 0.155/0.225/0.337; hard (Opus 8k+5 dB+10 % loss) 0.663. Set 2 audio (MMS) clean 0.192, hard 0.442.
* Cost: synthesis (Piper) ~75 s per 150 calls with 5 workers; ASR ~2.5 h for 3,764 new tasks on 5 workers (slower than the pilot rate on noisy conditions); MMS 124 calls ~48 min with 2 workers. Audio+features ~7 GB.
* 30 scenarios only: scenario diversity is limited by the dataset itself (scenario_id is 1-30).

## Phases 3-6: training, evaluation, ablations, error analysis (done)
* Text models were far slower than estimated (BiLSTM, ~3.8 s per batch of 32 at 256 tokens, not helped by threads); fixed with length-bucketed batches and capped epochs (2,000 samples/epoch, 5-8 epochs). 512-token ablation dropped for cost.
* The first audio run (patience 2, lr 2e-3) stopped at chance; re-run with lr 1e-3, 20 epochs, patience 6. Still near chance (AUROC 0.53-0.58). Diagnostic (`results/pace_diagnostic.json`): injected pace cue is AUROC 0.69 in labels but only 0.52 as measured speaking rate, because voice-to-voice speed variation (sd ~14 %) swamps the 9 % pace change.
* All 11 experiments x 3 seeds trained (21-48 min each, 3 CPU threads), 1 chain interruption (bash fork failure) fixed by re-running the resumable script.
* Results: H1 partly supported (damage comes from noise, not codecs), H2/H3/H4 not supported, H5 partly supported (see report, results/tables.md).
* Error analysis: 251 errors for F seed 0 across 1,736 call variants; 132 false alarms on scam-like normal calls; label noise and 60 s truncation visible in hand-checked cases.

## Phase 7: demo (done)
* Streamlit app, models T-asr-aug / A-aug / F-aug seed 0, 6 prepared test calls. Verified headlessly (streamlit.testing) and with a backup script. 3.2-4.5 s per 52-60 s call on an idle CPU.

## Phase 8: report, slides, docs (done)
* report/report.md (+ .docx/.pdf), report/slides.pptx (15 slides), docs/viva.md, docs/resume_lines.md, README.md, FINAL_SUMMARY.md.
