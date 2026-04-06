# YOLOv8N-SEG Segmentation Service

**Model:** `YOLOV8N_SEG-1.dxnn`
**Port:** 8004
**Task:** COCO instance segmentation (80 classes)
**Observed latency:** ~91ms (vs ~15ms for YOLOv8N)

---

## YOLOv8N vs YOLOv8N-SEG

Both use the same YOLOv8N backbone, trained on the same data. Detection accuracy — finding the right objects and classifying them — is the same.

YOLOv8N-SEG does one extra thing: it traces the outline of each detected object, returning a polygon that follows the actual shape — around arms, shoulders, the edge of a car — rather than just the rectangular box.

**In practice, YOLOv8N-SEG is strictly more useful.** It gives you everything YOLOv8N gives you plus the mask. A tumour example: YOLOv8N tells you where the tumour is. YOLOv8N-SEG tells you where it is and traces its exact boundary, so you can measure how big it is. The bounding box alone can't give you true size — it includes the rectangular space around the tumour, not just the tumour itself.

Most of the time you would just use YOLOv8N-SEG directly. The only reasons to use YOLOv8N instead:

- **Latency is critical** — YOLOv8N is ~6x faster on postprocessing
- **Output simplicity** — boxes are simpler to work with if you genuinely don't need the mask
- **Constrained hardware** — on very limited devices, the mask postprocessing overhead matters

The `polygon` field in the response is the traced outline, mapped back to original image coordinates.

### Why the extra latency

The NPU inference time is similar for both models. The ~91ms vs ~15ms difference comes from Python postprocessing: combining 32 mask coefficients with 32 prototype masks, sigmoid activation, upscaling each mask to 640×640, and running `cv2.findContours` to extract the polygon.

---

## When the Mask Matters

The bounding box from YOLOv8N is sufficient for most detection tasks. The segmentation mask matters when something downstream needs the shape of the object, not just its location.

**Medical imaging** — detecting a tumor. The box says where it is. The mask gives its exact shape and area, which is what you need to measure growth over time or define a radiation treatment boundary.

**Autonomous vehicles** — a pedestrian half-obscured by a lamppost. The box covers the lamppost too. The mask covers only the visible parts of the person, so the vehicle knows exactly how much road they occupy.

**Retail shelf analytics** — products touching each other have overlapping boxes. Masks separate each item so you can count accurately and detect out-of-stock gaps.

**Agriculture drones** — detecting diseased leaves. The mask gives exact area of diseased tissue per leaf, so you can calculate what percentage of the plant is affected.

**Two-stage pipelines** — detect a person with YOLOv8N-SEG, crop tightly to the torso region using the mask, then pass that crop to a classifier (e.g. safety vest check). Without the mask, the crop includes background clutter that degrades classifier accuracy.

That last pattern is the most relevant to this cluster: use YOLOv8N-SEG to isolate a region of interest, then pass the masked crop to another model running on the DX-M1.

---

## Output Format

```bash
curl -X POST http://localhost:8004/infer -F "file=@image.jpg"
```

```json
{
  "latency_ms": 91.3,
  "detections": [
    {
      "label": "person",
      "confidence": 0.8950,
      "box": {"x1": 5.2, "y1": 88.7, "x2": 537.4, "y2": 475.2},
      "polygon": [[277.9, 87.8], [276.8, 88.9], ...]
    }
  ]
}
```

`polygon` is the largest contour of the binary mask as `[[x, y], ...]` in original image coordinates. Empty list if no valid contour is found.

---

## Output Tensor Format

| Tensor | Shape | Contents |
|---|---|---|
| `output_tensors[0]` | `(1, 116, N)` | 4 box coords (cxcywh) + 80 class scores + 32 mask coefficients |
| `output_tensors[1]` | `(1, 32, mh, mw)` | Prototype masks |

Postprocessing steps:
1. Transpose output[0] → `(N, 116)`, split into boxes / class scores / mask coefs
2. Argmax class scores + NMS to select kept detections
3. Reconstruct masks: `kept_coefs @ proto.reshape(32, -1)` → sigmoid → resize to input dims
4. Crop masks to bounding box region
5. `cv2.findContours` on thresholded mask → largest contour → map to original image coords
