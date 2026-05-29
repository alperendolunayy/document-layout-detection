"""FastAPI endpoints for document layout detection."""

import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from docuvision.inference import LayoutDetector

app = FastAPI(
    title="DocuVision API",
    description="Document layout detection API powered by DETR",
    version="0.1.0",
)

# global detector instance
detector = None
CHECKPOINT_PATH = "checkpoints/last.ckpt"


def get_detector():
    """Load detector on first request."""
    global detector
    if detector is None:
        if not Path(CHECKPOINT_PATH).exists():
            return None
        detector = LayoutDetector(checkpoint_path=CHECKPOINT_PATH)
    return detector


@app.get("/health")
def health():
    """Check if service is running and model is loaded."""
    det = get_detector()
    return {
        "status": "healthy",
        "model_loaded": det is not None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), threshold: float = 0.5):
    """Detect layout elements in an uploaded document image.

    Args:
        file: uploaded image file
        threshold: confidence threshold (0.1 to 0.9)

    Returns:
        JSON with detected elements
    """
    det = get_detector()
    if det is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Check checkpoint path."},
        )

    # read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file."},
        )

    # clamp threshold
    threshold = max(0.1, min(0.9, threshold))

    # run detection
    detections = det.predict(image, threshold=threshold)

    return {
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections,
    }
