# vision-analyzer

A containerized FastAPI service running on **panda-worker** that accepts image and video uploads, extracts frames, and calls the YOLOv8N inference service on panda-control's NPU for object detection.

This is the first true containerized workload in the cluster — the NPU inference services (ports 8001–8004) run as host-native systemd processes and cannot be containerized due to DX-M1 kernel IPC constraints. vision-analyzer has no such constraint and runs as a normal K3s pod.

---

## Why NodePort (32005) Instead of a Direct Host Port

The NPU inference services are accessed directly on their host ports (8001–8004) because they run outside Kubernetes entirely. vision-analyzer runs *inside* K3s as a pod on panda-worker. To expose a pod to traffic from outside the cluster, Kubernetes requires a Service. A NodePort service binds a port in the 30000–32767 range on the node's IP and routes traffic into the pod.

```
External client → panda-worker:32005 → vision-analyzer pod (port 8005) → panda-control:8001 (YOLOv8N NPU)
```

This is the standard K3s pattern for external access to containerized workloads.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/analyze` | Analyze an image or video |
| GET | `/metrics` | Prometheus metrics |

### POST `/analyze`

**Query parameters:**
- `frame_interval` (int, 1–100, default 10) — analyze every Nth frame (video only)

**Request:** multipart form upload with field `file`

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

Any other extension is treated as an image.

**Response:**
```json
{
  "type": "image | video",
  "frames_analyzed": 1,
  "total_detections": 3,
  "latency_ms": 142.5,
  "frames": [
    {
      "frame_index": 0,
      "latency_ms": 38.2,
      "detections": [
        {
          "label": "person",
          "confidence": 0.91,
          "box": { "x1": 10.0, "y1": 20.0, "x2": 180.0, "y2": 400.0 }
        }
      ]
    }
  ]
}
```

---

## Usage

From any machine on the same network:

```bash
# Image
curl -X POST http://<panda-worker-ip>:32005/analyze -F "file=@photo.jpg"

# Video — analyze every 5th frame
curl -X POST "http://<panda-worker-ip>:32005/analyze?frame_interval=5" -F "file=@clip.mp4"
```

---

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `analysis_requests_total` | Counter | Total requests, labeled by `type` (image/video) |
| `frames_processed_total` | Counter | Total frames sent to yolov8n inference |
| `analysis_latency_ms` | Histogram | End-to-end latency per request |
| `inference_call_latency_ms` | Histogram | Per-frame round-trip latency to yolov8n |

Scraped via NodePort 32005 — target added to the `inference-scrape-config` secret in the `monitoring` namespace.

---

## Deployment

Runs on panda-worker via `nodeSelector: kubernetes.io/hostname: panda-worker`.

`YOLOV8N_INFERENCE_URL` is set to panda-control's LAN IP in `k8s/deployment.yaml` — `.local` mDNS hostnames are not resolvable from inside pods (CoreDNS only resolves cluster-internal DNS).

---

## Known Limitations

### Large video uploads fail with `curl: (52) Empty reply from server`

The entire video file is loaded into memory before frame extraction begins. Large files (tested: 68 MB `.mov`) exceed the pod's memory limit (512 Mi) and cause the connection to drop mid-upload with no HTTP response.

The pod memory limit is **512 Mi**. Memory is consumed in three stages:

1. **Full file loaded into memory** — `await file.read()` holds the entire upload in RAM
2. **Tempfile written to disk** — file bytes stay in memory while OpenCV reads from the tempfile
3. **Decoded frames accumulate** — each frame is `width × height × 3` bytes as a numpy array. At 1080p that's ~6 MB per frame

For a 10-second clip at 30fps with `frame_interval=10`, you get 30 analyzed frames × ~6 MB = ~180 MB just for frames, plus the raw file bytes on top. Rough safe ceiling: **~100–150 MB file size** at default `frame_interval=10` for 1080p video. Lower resolution or higher `frame_interval` gives more headroom.

**Workaround:**
```bash
# Trim to a shorter clip
ffmpeg -i video.mov -t 10 -c copy clip10s.mov
```

To raise the memory limit, update `resources.limits.memory` in `k8s/deployment.yaml`.
