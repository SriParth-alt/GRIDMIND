# Deployment image for the GRIDMIND live dashboard.
# Works on Hugging Face Spaces (Docker SDK), Render, Fly.io, or plain Docker.
#
# The image carries the precomputed simulation data (~1 MB) and the trained
# model (~80 KB) — the 700 MB raw ASHRAE dataset is never needed at runtime.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .

# CPU-only torch: ~200 MB instead of the multi-GB CUDA build. Inference on a
# 128-unit MLP needs no GPU, and free tiers cap image size.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "webapp/server.py"]
