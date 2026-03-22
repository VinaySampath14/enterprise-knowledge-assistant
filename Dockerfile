FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir -r /app/requirements.txt

# Copy runtime assets needed by the API.
COPY src /app/src
COPY config.yaml /app/config.yaml
COPY indexes /app/indexes

# Ensure logs path exists even on a fresh container filesystem.
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
