import os
import time
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from dx_engine import InferenceEngine

MODEL_PATH = os.environ["MOBILENETV2_MODEL_PATH"]

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
INFERENCE_LATENCY = Histogram(
    "inference_latency_ms", "NPU inference latency in milliseconds",
    buckets=[1, 2, 5, 10, 15, 20, 30, 50, 100]
)

_engine_ctx = None
engine = None
input_height = None
input_width = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine_ctx, engine, input_height, input_width
    _engine_ctx = InferenceEngine(MODEL_PATH)
    engine = _engine_ctx.__enter__()
    info = engine.get_input_tensors_info()
    input_height = info[0]["shape"][0]
    input_width = info[0]["shape"][1]
    yield
    _engine_ctx.__exit__(None, None, None)


app = FastAPI(lifespan=lifespan)


def preprocess(img_bgr, target_h, target_w):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return img_resized[np.newaxis, ...].astype(np.uint8)


def postprocess(outputs):
    # Model applies argmax internally — output is a single uint16 class ID
    return int(outputs[0][0])


@app.get("/health")
def health():
    return {"status": "ok", "model": "MobileNetV2"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "invalid image"})

    inp = preprocess(frame, input_height, input_width)

    t0 = time.perf_counter()
    outputs = engine.run([inp])
    latency_ms = (time.perf_counter() - t0) * 1000

    class_id = postprocess(outputs)

    REQUEST_COUNT.inc()
    INFERENCE_LATENCY.observe(latency_ms)

    return {"latency_ms": round(latency_ms, 2), "class_id": class_id}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
