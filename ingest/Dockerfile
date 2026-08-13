FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/models TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# One resolve pass, with the CPU wheel index available as an extra source.
# Splitting this into two pip calls makes the resolver thrash.
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
COPY ingest.py worker.py ./
ENTRYPOINT ["python", "ingest.py"]
CMD ["/data"]
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
