# Playwright's image supplies Chromium system dependencies for rendered extraction.
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

WORKDIR /app

# Podcast mode uses ffprobe for duration limits and ffmpeg for bounded,
# speech-optimized chunks before OpenAI diarized transcription.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY back-end /app/back-end
COPY data/fact_opinion/processed/fact_opinion_classifier.pkl /app/data/fact_opinion/processed/fact_opinion_classifier.pkl

# Run the public API as the image's unprivileged browser user.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app/back-end
USER pwuser

CMD ["sh", "-c", "uvicorn home:app --host 0.0.0.0 --port ${PORT:-8000}"]
