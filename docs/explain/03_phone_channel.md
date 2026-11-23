# Phone-line simulation
**What it does.** Real calls are not studio audio. We degrade each call with: added noise at a chosen signal-to-noise ratio (SNR), squeezing to 8 kHz bandwidth, a telephone codec (G.711 mu-law/A-law, GSM, AMR-NB, Opus at 8-12 kbps) and random dropped 20 ms packets. `apply_channel(audio, config, seed)` always gives the same output for the same seed.
**Why.** Compression and noise change how speech recognition works, so a text detector that looks great on clean transcripts may fail on a real line. Measuring that gap is the point of the project.
**Alternative.** Record real calls (not possible: privacy, and no public Indian scam-call audio set exists to our knowledge).
