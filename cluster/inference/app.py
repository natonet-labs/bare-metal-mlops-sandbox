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

MODEL_PATH = "/home/delta/dx-all-suite/workspace/res/models/models-2_2_1/YoloV8N.dxnn"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
INFERENCE_LATENCY = Histogram(
    "inference_latency_ms", "NPU inference latency in milliseconds",
    buckets=[5, 10, 15, 20, 30, 50, 100, 200]
)
DETECTION_COUNT = Histogram(
    "detections_per_frame", "Number of detections returned per inference",
    buckets=[0, 1, 2, 5, 10, 20, 50]
)

_engine_ctx = None
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine_ctx, engine
    _engine_ctx = InferenceEngine(MODEL_PATH)
    engine = _engine_ctx.__enter__()
    yield
    _engine_ctx.__exit__(None, None, None)


app = FastAPI(lifespan=lifespan)


def letterbox(img, target_size=640):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_right = target_size - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, pad_left, pad_top


def postprocess(outputs, orig_shape, scale, pad_left, pad_top):
    pred = outputs[0].reshape(84, -1).T
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]
    mask = confidences >= CONF_THRESHOLD
    boxes_cxcywh = pred[:, :4][mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    if len(boxes_cxcywh) == 0:
        return [], [], []
    orig_h, orig_w = orig_shape[:2]
    x1 = np.clip((boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2 - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    keep = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        nms_idx = cv2.dnn.NMSBoxes(
            boxes[idx].tolist(), confidences[idx].tolist(),
            CONF_THRESHOLD, IOU_THRESHOLD
        )
        if len(nms_idx) > 0:
            keep.extend(idx[np.array(nms_idx).flatten()])
    return boxes[keep], confidences[keep], class_ids[keep]


@app.get("/health")
def health():
    return {"status": "ok", "model": "YoloV8N"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "invalid image"})

    padded, scale, pad_left, pad_top = letterbox(frame)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    inp = rgb[np.newaxis, ...].astype(np.uint8)

    t0 = time.perf_counter()
    outputs = engine.run([inp])
    latency_ms = (time.perf_counter() - t0) * 1000

    boxes, scores, class_ids = postprocess(outputs, frame.shape, scale, pad_left, pad_top)

    REQUEST_COUNT.inc()
    INFERENCE_LATENCY.observe(latency_ms)
    DETECTION_COUNT.observe(len(boxes))

    detections = [
        {
            "label": COCO_CLASSES[int(cls_id)],
            "confidence": round(float(score), 4),
            "box": {
                "x1": round(float(box[0]), 1),
                "y1": round(float(box[1]), 1),
                "x2": round(float(box[2]), 1),
                "y2": round(float(box[3]), 1),
            },
        }
        for box, score, cls_id in zip(boxes, scores, class_ids)
    ]

    return {"latency_ms": round(latency_ms, 2), "detections": detections}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
