# Viva sheet: short spoken-style answers

**Why a BiGRU (and BiLSTM) and not a transformer?**
The brief asks for a CNN or RNN backbone, and on this machine they are what trains in minutes on a CPU. For the audio, convolutions find local patterns in the spectrogram and the GRU follows how they change over 30 seconds. A transformer is the stronger text baseline (we planned a distilled multilingual one, T-rob, as a Colab notebook), but we did not run it, so I cannot claim it would or would not do better.

**Why simulate the phone line?**
Real calls are squeezed to 8 kHz, compressed by a codec, noisy and lose packets, and speech recognition then makes mistakes. Benchmarks usually test on clean text, so they hide that. By degrading the same call in controlled steps (14 conditions) we can say which part of the line hurts and by how much. In our data, codecs and packet loss barely mattered (recognition error stayed at 14-16 % against 13.5 % clean); noise at 10 dB and 5 dB raised it to 23 % and 34 %, and the combined "hard" condition to 66 %.

**What happens when speech recognition is wrong?**
The text branch sees wrong words. With recognition error up to about 34 % the detector held up (AUROC 0.96 clean, 0.91 at 5 dB noise), because a fraud call contains many cues and a few words wrong do not erase them. At 66 % error (the "hard" condition) AUROC fell to 0.80 and F1 from 0.88 to 0.68. Names and Hinglish words are the words most often garbled, and that is not what decides the verdict.

**Why is synthetic audio a fair test, and what are its limits?**
It is fair in one way: we know the exact text and the exact degradation, so every loss is attributable. It is not fair in others: the voices are clean text-to-speech, the conversations are themselves LLM-written, the Roman-script Hinglish is read with English pronunciation, and the audio cue for "fast-paced" is injected by us from dataset labels and is weak next to natural voice-to-voice speed differences (measured: speaking-rate AUROC 0.52). Results may not transfer to real calls; that is the first limitation in the report.

**How is time-to-detection measured?**
At each 10-second step the model gives a fraud score. We choose a threshold on validation data so that only 5 % of normal calls ever cross it at any step. On test calls, time-to-detection is the number of seconds until a fraud call's score first crosses that threshold; calls that never cross are counted as missed. We then plot the share of fraud calls flagged against seconds into the call.

**Why are splits made by session?**
Each `session_id` in ICFD-31k holds 10 conversations of the same scenario. If some were in training and others in test, the model could recognise the scenario's wording and look better than it is. A test checks that no session appears in more than one split.

**What does the unseen-TTS test show?**
We trained only on Piper voices and tested on calls rendered by a different system (MMS-TTS). The text-based models lost little (F1 within about 0.02 on clean audio); the audio-only model trained on clean audio collapsed (F1 0.67 to 0.23), showing audio branches can latch onto the voice system. Limit: MMS has only one voice, so "voices" there are pitch/speed variants, and the audio branch was close to chance anyway.

**Did the acoustic branch help?**
No. In our data the audio-only branch was near chance (AUROC about 0.55), and fusion matched the text-only model; with five seeds the paired differences were within about 0.02 AUROC and every confidence interval included zero. The early-detection advantage seen with three seeds reversed with five, so it was seed luck. We report that as measured and explain it by the weak acoustic cue in synthetic speech.
