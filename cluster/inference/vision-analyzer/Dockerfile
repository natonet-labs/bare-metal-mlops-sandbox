FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart opencv-python httpx numpy prometheus-client
COPY app.py .
EXPOSE 8005
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
