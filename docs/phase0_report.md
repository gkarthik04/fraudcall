# Phase 0 report: day-one checks

All numbers come from `results/phase0_tts_asr.json`, `results/phase0_codecs.json` and commands run on this machine.

## 1. Resources
| Item | Value |
|---|---|
| CPU | Intel Core i5-1235U, 10 cores / 12 threads (2 P + 8 E) |
| RAM | 15.7 GB |
| Free disk | 20.4 GB on C: |
| GPU | none (`nvidia-smi` not found; `torch.cuda.is_available()` = False, torch 2.11.0+cpu) |
| Python | 3.11.0 |
| ffmpeg | 2025-03-27 gyan.dev full build. `sox` is not installed and not needed: ffmpeg covers every channel step |

Everything is designed for CPU. Disk is the tight resource, so audio is stored at 8 kHz mono 16-bit (clean reference at 16 kHz).

## 2. Data: ICFD-31k (rishia2220/icfd-31k)
* Used: `source_conversations/*.jsonl.zst` (full conversations). The `streaming_chunks/` parquet files (3-second cumulative text chunks) are not used because our chunks are cut from audio.
* Schema, one row per conversation: `transcript` (list of `{speaker, text, timestamp_end}`), `multimodal_analysis`, `key_entities`, `chunk_level_analysis`, `final_verdict` (YES/NO), `scam_outcome`, `session_id`, `agent_persona`, `customer_persona`, `scenario {scenario_id, description, case_type, policy}`, `release_metadata {split, domain, conversation_uid, ...}`.
* Counts: train 21,000, validation 4,500, test 4,500, cross_domain 1,000 conversations.
* **Licence.** The Hugging Face card says `license: other` with no text. The GitHub repository (SPELLAILab/ICFD-31k) has `LICENSE` (MIT, code) and `LICENSE-DATA.md`, a restricted *defensive-use* licence. Permitted: academic research on fraud detection, defensive AI, education, benchmarking. Prohibited: generating scam scripts, offensive tools, commercial deployment without written permission. Citation required. **A non-commercial course project building a defensive detector is a permitted use.** We write no new scam scripts: audio is rendered only from existing transcripts and is kept private.
* The dataset is itself synthetic (LLM-written conversations): no real recordings or real PII.

### Findings that change the design
1. **The official splits cannot be used as given.** `final_verdict` per official split:

| split | YES (fraud) | NO |
|---|---|---|
| train | 6,481 | 14,519 |
| validation | 4,474 | 26 |
| test | 4,477 | 23 |
| cross_domain | 961 | 39 |

   Validation and test are about 99.5% fraud, so they cannot measure false alarms. Almost all normal calls are in train. We therefore **pool train+validation+test and re-split ourselves** (70/15/15 by `session_id`, stratified by label) from a balanced sample.
2. **`session_id` is not unique per conversation.** Each id covers exactly 10 conversations of one scenario (2,100 ids for 21,000 train rows). Splitting by conversation would leak scenario wording across splits, so we split by `session_id`. Train, validation and test share no `session_id` (checked, 0 overlap), but cross_domain shares 100 ids with train and 100 with validation, so cross_domain is not used.
3. **Label noise exists.** 582 conversations with case_type "Clear Normal" have verdict YES, and 21 "Clear Fraud" have verdict NO. We use `final_verdict` as released and list this in the limitations.
4. **Text is Roman-script English/Hinglish**, about 13 turns and 223 words per conversation. Only 2 train turns contain Devanagari. "Hindi" in this project therefore means Roman-script Hinglish.
5. **`timestamp_end` is unreliable** (some values are strings such as `"2.61*60"`), so it is not used. Call timing comes from the synthesized audio.

## 3. TTS benchmark (CPU, 4 threads)
Test clips: 5-turn two-speaker conversations (Roman Hinglish and English from ICFD; a hand-written 4-turn Devanagari Hindi one). Speed is seconds of audio per second of compute. The first Piper call includes warm-up.

| Model | Languages | Speed | Licence | Notes |
|---|---|---|---|---|
| Piper `en_US-libritts_r-medium` | English, 904 speakers | 3.1x (cold) to 14.2x | voice data CC BY 4.0; engine GPL-3.0-or-later | Many built-in voices, fast. Reads Roman Hinglish with English phonetics, so words like "Achha" are mispronounced. **Chosen as TTS Set 1.** |
| Piper `hi_IN-pratham-medium` | Hindi (Devanagari), 1 speaker | 14.7x | CC BY-NC-SA 4.0 | Clear, but ICFD has no Devanagari text. Not used. |
| MMS-TTS `facebook/mms-tts-eng` | English, 1 speaker | 0.81x to 0.83x | CC BY-NC 4.0 | Intelligible but ~15x slower than Piper. **Chosen as TTS Set 2** (test split only). |
| MMS-TTS `facebook/mms-tts-hin` | Hindi (Devanagari), 1 speaker | 0.87x | CC BY-NC 4.0 | Same speed problem. Not used. |

XTTS-v2 and Indic Parler-TTS were not tried: even the small VITS-class MMS model runs below real time on this CPU and both are much larger, so they fail the "acceptable CPU speed" rule.

Consequences: Set 1 = Piper libritts_r with a pool of its built-in speakers. Set 2 = MMS-TTS-eng on the test split only. MMS has a single voice, so Set 2 "voices" are rate/pitch/noise variants of one voice, not different speakers; this limit is repeated wherever H5 is reported. **Telugu is dropped**: no TTS/ASR support was found or tested.

## 4. Codecs (`results/phase0_codecs.json`)
All six round-trip correctly: G.711 mu-law, G.711 A-law, GSM (libgsm), Opus 8 kbps and 12 kbps (libopus), AMR-NB 12.2 kbps (libopencore_amrnb). Nothing is missing.

## 5. Speech recognition (faster-whisper, int8, 6 threads, greedy, `language="en"`)
WER against the known transcript, on clean Piper/MMS audio (no channel degradation yet):

| clip | tiny WER | base WER |
|---|---|---|
| Piper, Hinglish-Roman, 18.8 s | 0.107 | 0.067 |
| Piper, English, 27.8 s | 0.230 | 0.195 |
| MMS, Hinglish-Roman | 0.107 | 0.093 |
| MMS, English | 0.264 | 0.218 |
| Piper, Hindi Devanagari | 1.156 | 1.067 |
| MMS, Hindi Devanagari | 1.178 | 1.156 |

Speed: tiny 11-14x real time, base 7-10x on English clips. Whisper tiny/base cannot transcribe Hindi (base emitted Urdu script, WER > 1). Even English WER is 7-26% on clean synthetic audio, mostly Indian names and Hinglish words. These are 3 clips, so they indicate magnitude, not precise rates; Phase 1 measures WER on the full pilot. **Decision: faster-whisper `base`, int8, forced English, on Roman Hinglish.**

## 6. Scale decision
Estimated per call (60 s cap): Piper ~6 s, each channel variant ~1 s, ASR (base) ~8 s. Plan: about 600 conversations, 1 clean render each, ~2 random channel variants per train/val call, a ~14-condition fixed grid on test calls, plus Set 2 on test calls. That is roughly 3,000 ASR runs (4-6 h, parallel workers), ~1.5 h Piper, ~2 h MMS. Total is well under 24 h and no stage needs Colab.
