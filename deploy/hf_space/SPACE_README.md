---
title: FraudCall
emoji: 📞
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Robust fraud-call detection on phone speech (research prototype)
---

# FraudCall: robust fraud-call detection on phone speech

Research prototype. It scores a phone call for fraud as it unfolds, ten seconds at a time, from recognised speech and the sound of the line, and was tested across 14 simulated phone-channel conditions.

**Limitation:** the system is trained and evaluated on synthetic speech generated from dataset transcripts, so results may not transfer to real calls. It is not a security product.

* Code, report and full results: https://github.com/gkarthik04/fraudcall
* Data: ICFD-31k transcripts (restricted defensive-use licence: academic and defensive research only; no scam-script generation, no commercial deployment). The six sample calls are synthetic speech rendered from existing dataset transcripts; no real person's voice was cloned.
* Uploaded audio is processed in memory and never stored.
* Speech synthesis: Piper `en_US-libritts_r-medium` (CC BY 4.0 voice data). Speech recognition: faster-whisper `base` (MIT).
