from pydantic import BaseModel
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch


class Detection(BaseModel):
    label: str
    confidence: float
    location: tuple[float, float, float, float]


class CoderDetector:
    def __init__(self) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-tiny"
        ).eval()

    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: float,
    ) -> list[Detection]:
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.T.shape[1:]])
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        detections: list[Detection] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = tuple(round(i, 2) for i in box.tolist())
            detections.append(
                Detection(
                    label=self.model.config.id2label[label.item()],
                    confidence=round(score.item(), 3),
                    location=box,
                )
            )
        return detections

    def is_coder(
        self,
        image: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[bool, float]:
        detections = self.detect_objects(
            image, confidence_threshold=confidence_threshold
        )
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        person_present = False
        computer_present = False
        computer_labels = {
            "laptop",
            "mouse",
            "keyboard",
            "tv",
        }
        confidence = 1.0
        for detection in detections:
            if not person_present and detection.label == "person":
                person_present = True
                confidence *= detection.confidence
            if not computer_present and detection.label in computer_labels:
                computer_present = True
                confidence *= detection.confidence
            if person_present and computer_present:
                if confidence < confidence_threshold:
                    return False, 1.0 - confidence
                return True, confidence
        return False, confidence
