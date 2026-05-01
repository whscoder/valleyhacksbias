FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY back-end /app/back-end

ENV PYTHONUNBUFFERED=1
WORKDIR /app/back-end

CMD ["sh", "-c", "uvicorn home:app --host 0.0.0.0 --port ${PORT:-8000}"]
