# CPU-only serving image. Inference on 64x64 patches needs no GPU, and a CPU wheel of
# PyTorch keeps the image a fraction of the CUDA size and portable to any host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[serve]"
COPY scripts ./scripts

# Non-root: a compromised request handler should not own the container filesystem.
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Weights and processed data are mounted at run time, never baked into the image.
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
