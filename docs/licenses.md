# Licences of every dataset, model and tool used

| Component | Source | Licence | How we respect it |
|---|---|---|---|
| ICFD-31k data | huggingface.co/datasets/rishia2220/icfd-31k, github.com/SPELLAILab/ICFD-31k | Card: `other`. `LICENSE-DATA.md`: restricted defensive-use licence (academic/defensive research and education allowed; scam-script generation, offensive tools and commercial deployment prohibited; citation required) | Non-commercial course research on a detector; we cite the paper; raw data, generated audio and checkpoints are not published |
| ICFD-31k code | same GitHub repo | MIT | Not copied |
| Piper TTS engine (`piper-tts` 1.8.0) | github.com/OHF-voice/piper1-gpl | GPL-3.0-or-later | Used as an unmodified library; we do not distribute it |
| Piper voice `en_US-libritts_r-medium` | huggingface.co/rhasspy/piper-voices | CC BY 4.0 (LibriTTS-R data, OpenSLR 141) | Attribution here; model not redistributed |
| Piper voice `hi_IN-pratham-medium` (benchmark only) | rhasspy/piper-voices | CC BY-NC-SA 4.0 (per model card) | Used only in the Phase 0 benchmark |
| MMS-TTS `facebook/mms-tts-eng`, `-hin` | Hugging Face (Meta) | CC BY-NC 4.0 | Non-commercial course use only; Set 2 audio is not published |
| faster-whisper 1.2.1 | github.com/SYSTRAN/faster-whisper | MIT | |
| Whisper tiny/base (CTranslate2 conversions `Systran/faster-whisper-*`) | Hugging Face | MIT (card); original `openai/whisper-base` is Apache-2.0 per its card | |
| XLM-RoBERTa tokenizer `FacebookAI/xlm-roberta-base` | Hugging Face | MIT | Tokenizer only |
| DistilBERT multilingual `distilbert/distilbert-base-multilingual-cased` (optional T-rob) | Hugging Face | Apache-2.0 | |
| Noise | generated in code (white, pink, brown, babble made from our own TTS audio) | n/a (no external noise corpus used) | |
| PyTorch, scikit-learn, jiwer, librosa, streamlit, python-pptx, etc. | PyPI | BSD / MIT / Apache-2.0 (see each package) | |
| ffmpeg (gyan.dev full build) | ffmpeg.org | GPL (full build includes libgsm, opencore-amr, libopus) | Used as an external command-line tool |
