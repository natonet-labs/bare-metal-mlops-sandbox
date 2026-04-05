import io
import time
import numpy as np
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from dx_engine import InferenceEngine

import os
MODEL_PATH = os.environ["SCRFD_MODEL_PATH"]
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
NUM_KEYPOINTS = 5
KEYPOINT_NAMES = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
INFERENCE_LATENCY = Histogram(
    "inference_latency_ms", "NPU inference latency in milliseconds",
    buckets=[5, 10, 15, 20, 30, 50, 100, 200]
)
FACE_COUNT = Histogram(
    "faces_per_frame", "Number of faces detected per inference",
    buckets=[0, 1, 2, 3, 5, 10, 20]
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
    input_height = info[0]["shape"][1]
    input_width = info[0]["shape"][2]
    yield
    _engine_ctx.__exit__(None, None, None)


app = FastAPI(lifespan=lifespan)


def letterbox(img, target_h, target_w):
    h, w = img.shape[:2]
    gain = min(target_h / h, target_w / w)
    new_w = int(round(w * gain))
    new_h = int(round(h * gain))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = int(round((target_h - new_h) / 2 - 0.1))
    pad_left = int(round((target_w - new_w) / 2 - 0.1))
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left
    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img, gain, pad_top, pad_left


def postprocess(output_tensors, gain, pad_top, pad_left, orig_h, orig_w):
    num_anchors = 2

    buckets = {}
    for t in output_tensors:
        if t.ndim != 3:
            continue
        b, n, c = t.shape
        if b != 1:
            continue
        entry = buckets.setdefault(n, {})
        if c == 1:
            entry["score"] = t
        elif c == 4:
            entry["bbox"] = t
        elif c == 10:
            entry["kps"] = t

    triplets = [
        (
            input_width // max(1, int(round(np.sqrt(n // num_anchors)))),
            d["score"], d["bbox"], d["kps"],
        )
        for n, d in buckets.items()
        if {"score", "bbox", "kps"} <= d.keys()
    ]

    all_boxes, all_scores, all_keypoints = [], [], []

    for stride, score_t, bbox_t, kps_t in triplets:
        score = score_t.reshape(-1)
        bbox = bbox_t.reshape(-1, 4)
        kps = kps_t.reshape(-1, NUM_KEYPOINTS * 2)

        n = score.size
        if n == 0:
            continue

        hw = max(1, int(round(np.sqrt(n // num_anchors))))
        loc = np.arange(n) // num_anchors
        gx, gy = loc % hw, loc // hw
        cx, cy = gx * stride, gy * stride

        x1 = cx - bbox[:, 0] * stride
        y1 = cy - bbox[:, 1] * stride
        x2 = cx + bbox[:, 2] * stride
        y2 = cy + bbox[:, 3] * stride
        boxes_xywh = np.column_stack([x1, y1, x2 - x1, y2 - y1])

        kx = cx[:, None] + kps[:, 0::2] * stride
        ky = cy[:, None] + kps[:, 1::2] * stride
        keypoints = np.stack((kx, ky), axis=-1).reshape(len(score), -1)

        all_boxes.append(boxes_xywh)
        all_scores.append(score)
        all_keypoints.append(keypoints)

    if not all_boxes:
        return []

    boxes_xywh = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    keypoints_all = np.vstack(all_keypoints)

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(), SCORE_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) == 0:
        return []

    keep = np.array(indices).reshape(-1)
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    keypoints_all = keypoints_all[keep]

    # Convert xywh → x1y1x2y2 and map back to original image coordinates
    x1 = np.clip((boxes_xywh[:, 0] - pad_left) / gain, 0, orig_w - 1)
    y1 = np.clip((boxes_xywh[:, 1] - pad_top) / gain, 0, orig_h - 1)
    x2 = np.clip((boxes_xywh[:, 0] + boxes_xywh[:, 2] - pad_left) / gain, 0, orig_w - 1)
    y2 = np.clip((boxes_xywh[:, 1] + boxes_xywh[:, 3] - pad_top) / gain, 0, orig_h - 1)

    results = []
    for i in range(len(keep)):
        kps = keypoints_all[i]
        landmarks = {}
        for k, name in enumerate(KEYPOINT_NAMES):
            kx = float(np.clip((kps[k * 2] - pad_left) / gain, 0, orig_w - 1))
            ky = float(np.clip((kps[k * 2 + 1] - pad_top) / gain, 0, orig_h - 1))
            landmarks[name] = [round(kx, 1), round(ky, 1)]

        results.append({
            "confidence": round(float(scores[i]), 4),
            "box": {
                "x1": round(float(x1[i]), 1),
                "y1": round(float(y1[i]), 1),
                "x2": round(float(x2[i]), 1),
                "y2": round(float(y2[i]), 1),
            },
            "keypoints": landmarks,
        })

    return results


@app.get("/health")
def health():
    return {"status": "ok", "model": "SCRFD500M"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "invalid image"})

    orig_h, orig_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    letterboxed, gain, pad_top, pad_left = letterbox(rgb, input_height, input_width)

    t0 = time.perf_counter()
    outputs = engine.run([letterboxed])
    latency_ms = (time.perf_counter() - t0) * 1000

    faces = postprocess(outputs, gain, pad_top, pad_left, orig_h, orig_w)

    REQUEST_COUNT.inc()
    INFERENCE_LATENCY.observe(latency_ms)
    FACE_COUNT.observe(len(faces))

    return {"latency_ms": round(latency_ms, 2), "faces": faces}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
