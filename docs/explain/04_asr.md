# Speech recognition (ASR)
**What it does.** faster-whisper (the small "base" Whisper model, 8-bit, CPU) turns degraded audio back into text with word timestamps. We run it once per call and cut the text at 10, 20, ... seconds, which equals what a live recogniser would have heard so far.
**Why this model.** It is the smallest model with acceptable accuracy that runs on a laptop. Errors (misheard Hinglish words, names) are part of what we measure, so we keep the model small and report the word error rate (WER) per condition.
**Limit.** Whisper tiny/base cannot transcribe Hindi in Devanagari at all (we measured WER above 1), which is one reason the project uses Roman-script Hinglish.
