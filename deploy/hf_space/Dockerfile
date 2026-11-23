FROM python:3.11-slim

# ffmpeg decodes uploaded audio and applies the phone codecs
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# fetch the speech recogniser and tokenizer files at build time so start-up is quick (no weights of the large XLM-R model)
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Systran/faster-whisper-base'); \
snapshot_download('FacebookAI/xlm-roberta-base', allow_patterns=['*.json', 'sentencepiece.bpe.model'])"

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
