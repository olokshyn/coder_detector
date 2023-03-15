from fastapi import FastAPI, UploadFile
import numpy as np
import cv2
from pydantic import BaseModel

from coder_detector.ml_pipeline.serving import CoderDetector


app = FastAPI()
detector = CoderDetector()


class IsCoderResponse(BaseModel):
    is_coder: bool
    confidence: float


@app.post("/is_coder")
async def is_coder(image: UploadFile, confidence_threshold: float = 0.2) -> IsCoderResponse:
    raw_data = await image.read()
    image = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR)
    is_coder, confidence = detector.is_coder(
        image, confidence_threshold=confidence_threshold
    )
    return IsCoderResponse(
        is_coder=is_coder,
        confidence=confidence,
    )
