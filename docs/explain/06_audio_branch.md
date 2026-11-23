# Audio branch: CNN + BiGRU
**What it does.** The last 30 seconds of audio become a picture (log-mel spectrogram: time across, pitch bands up). Three small convolution blocks find local sound patterns, a bidirectional GRU follows how they change over time, and attention pooling gives a fraud probability. It hears pace, pauses, energy and line quality that text loses.
**Why.** CNNs are good at spectrogram patterns and a GRU at their order in time. Time is shrunk 4x by pooling so a 30 s window stays cheap on CPU.
