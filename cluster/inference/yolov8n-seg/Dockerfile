FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart opencv-python-headless numpy prometheus-client
COPY app.py .
# NOTE: DX-M1 inference cannot run in a container — dxrtd uses /dev/dxrt0 as
# a kernel-level IPC bus that container isolation breaks even with hostPID and
# privileged mode. This image packages the service code only. The running
# service uses the host-native systemd pattern (see node01-single-node-readiness-plan.md).
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
