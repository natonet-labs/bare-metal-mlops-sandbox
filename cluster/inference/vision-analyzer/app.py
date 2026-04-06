import os
import time
import tempfile
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

YOLOV8N_INFERENCE_URL = os.environ["YOLOV8N_INFERENCE_URL"]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Prometheus metrics
REQUEST_COUNT = Counter(
    "analysis_requests_total", "Total analysis requests", ["type"]
)
FRAMES_PROCESSED = Counter(
    "frames_processed_total", "Total frames sent to inference"
)
ANALYSIS_LATENCY = Histogram(
    "analysis_latency_ms", "End-to-end analysis latency in milliseconds",
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000]
)
INFERENCE_CALL_LATENCY = Histogram(
    "inference_call_latency_ms", "Per-frame yolov8n call latency in milliseconds",
    buckets=[5, 10, 20, 50, 100, 200, 500]
)

app = FastAPI()


def is_video(filename: str, content_type: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in VIDEO_EXTENSIONS or (content_type or "").startswith("video/")


def extract_frames(video_bytes: bytes, frame_interval: int):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames, indices = [], []
    try:
        cap = cv2.VideoCapture(tmp_path)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_interval == 0:
                frames.append(frame)
                indices.append(idx)
            idx += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames, indices


async def run_inference(client: httpx.AsyncClient, frame: np.ndarray) -> dict:
    _, encoded = cv2.imencode(".jpg", frame)
    t0 = time.perf_counter()
    r = await client.post(
        f"{YOLOV8N_INFERENCE_URL}/infer",
        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    INFERENCE_CALL_LATENCY.observe(latency_ms)
    FRAMES_PROCESSED.inc()
    return r.json()


@app.get("/health")
def health():
    return {"status": "ok", "service": "vision-analyzer"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    frame_interval: int = Query(default=10, ge=1, le=100, description="Analyze every Nth frame (video only)"),
):
    contents = await file.read()
    t0 = time.perf_counter()

    if is_video(file.filename, file.content_type):
        frames, frame_indices = extract_frames(contents, frame_interval)
        input_type = "video"
        if not frames:
            return JSONResponse(status_code=400, content={"error": "no frames extracted from video"})
    else:
        img_array = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "invalid image"})
        frames = [frame]
        frame_indices = [0]
        input_type = "image"

    frame_results = []
    async with httpx.AsyncClient(timeout=30) as client:
        for frame, idx in zip(frames, frame_indices):
            result = await run_inference(client, frame)
            frame_results.append({
                "frame_index": idx,
                "latency_ms": result.get("latency_ms"),
                "detections": result.get("detections", []),
            })

    total_latency = (time.perf_counter() - t0) * 1000
    total_detections = sum(len(f["detections"]) for f in frame_results)

    REQUEST_COUNT.labels(type=input_type).inc()
    ANALYSIS_LATENCY.observe(total_latency)

    return {
        "type": input_type,
        "frames_analyzed": len(frames),
        "total_detections": total_detections,
        "latency_ms": round(total_latency, 2),
        "frames": frame_results,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
