FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY back-end /app/back-end

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app/back-end
USER pwuser

CMD ["sh", "-c", "uvicorn home:app --host 0.0.0.0 --port ${PORT:-8000}"]
