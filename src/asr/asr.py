"""Speech recognition (faster-whisper, int8, CPU) with word timestamps, plus WER scoring.

We transcribe each call ONCE with word timestamps and cut the transcript at any time t by keeping the
words that ended by t. This is much cheaper than re-transcribing every cumulative 10 s prefix and gives
(almost) the same text a streaming recogniser would have produced.
"""
import re
import numpy as np
import jiwer
from scipy.signal import resample_poly

_NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])


def normalize_text(s):
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    return _NORM(s)


def wer(ref, hyp):
    """Word error rate after lower-casing and punctuation removal. Empty reference -> nan."""
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return float("nan")
    if not h:
        return 1.0
    return float(jiwer.wer(r, h))


class ASR:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel
        a = cfg["asr"]
        self.model = WhisperModel(a["model"], device="cpu", compute_type=a["compute_type"], cpu_threads=a["threads"])
        self.lang = a["language"]

    def transcribe(self, audio, sr):
        """Return list of (word, start, end) for float32 mono audio at sample rate sr."""
        if sr != 16000:
            g = np.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        segs, _ = self.model.transcribe(audio.astype(np.float32), language=self.lang, beam_size=1,
                                        word_timestamps=True, condition_on_previous_text=False,
                                        vad_filter=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append((w.word.strip(), round(float(w.start), 2), round(float(w.end), 2)))
        return words


def words_until(words, t):
    return " ".join(w for w, _, e in words if e <= t and w)
