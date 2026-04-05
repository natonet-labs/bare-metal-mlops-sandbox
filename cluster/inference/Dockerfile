FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi "uvicorn[standard]" numpy opencv-python-headless prometheus-client python-multipart

# dx_engine Python package (copied from host venv)
COPY dx_engine/ /usr/local/lib/python3.12/dist-packages/dx_engine/

# DX-RT runtime libraries (required by dx_engine/_pydxrt.so)
COPY libdxrt.so /usr/local/lib/libdxrt.so
COPY libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so.1
RUN ldconfig

# Model and application
COPY models/ /models/
COPY app.py /app.py

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
