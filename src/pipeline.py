"""
Two-Stage Plant Disease Diagnostic Pipeline Coordinator.
Orchestrates Leaf Localization -> DenseNet169 Classification -> OpenCV Visual Overlay.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from .detector import LeafDetector
from .classifier import DiseaseClassifier
from .visualizer import Visualizer


# Neutral, safety-oriented copy used in place of agronomic treatment advice
# whenever a result isn't a trustworthy diagnosis. These strings are what
# gets shown to the user under `result["advice"]` for those cases — see
# PlantDiagnosticPipeline.run().
_UNTRAINED_ADVICE = (
    "No model checkpoint is loaded. This classifier's head is randomly "
    "initialized and untrained — its output carries no diagnostic meaning "
    "whatsoever. Do not act on it."
)
_LOW_CONFIDENCE_ADVICE = (
    "The model's top prediction is below the configured confidence "
    "threshold. This usually means the image is a poor match for anything "
    "the model was trained on (wrong crop, unfamiliar condition, poor crop/"
    "lighting, or an out-of-distribution photo). Treat this as 'no reliable "
    "classification' rather than a diagnosis."
)


class PlantDiagnosticPipeline:
    """
    End-to-End Plant Disease Diagnostic Pipeline.
    Integrates leaf localization with fine-tuned DenseNet-169 classification
    and visual overlay rendering.
    """

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        classifier_weights_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        detector_confidence: float = 0.35,
        device: Optional[str] = None,
    ):
        """
        Initialize the two-stage diagnostic pipeline.

        Args:
            yolo_model_path: Path to YOLO weights for localization (optional).
            classifier_weights_path: Path to trained DenseNet169 weights (.pth).
            class_mapping_path: Path to class_mapping.json.
            detector_confidence: Confidence threshold for leaf detection.
            device: 'cuda' or 'cpu'.
        """
        print("[PlantDiagnosticPipeline] Initializing pipeline components...")
        self.detector = LeafDetector(
            yolo_model_path=yolo_model_path,
            confidence_threshold=detector_confidence
        )
        self.classifier = DiseaseClassifier(
            weights_path=classifier_weights_path,
            class_mapping_path=class_mapping_path,
            device=device
        )
        self.visualizer = Visualizer()
        print("[PlantDiagnosticPipeline] Pipeline initialized successfully.")

    def run(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 5,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute full two-stage diagnostic workflow on an input image.

        Args:
            image_input: Filepath, PIL Image, or RGB numpy array.
            top_k: Number of highest ranking classes to compute.
            confidence_threshold: If provided, top-1 predictions with
                confidence below this value are treated as "low_confidence"
                (no health badge, no treatment advice) instead of a
                diagnosis, even if the model is trained. Pass None (default)
                to disable this check — e.g. for scripts/evaluate.py, which
                needs raw predictions regardless of confidence.

        Returns:
            Dictionary containing:
                - 'annotated_image': np.ndarray (RGB) with bounding box and status badge
                - 'crop_image': np.ndarray (RGB) of localized leaf ROI
                - 'probability_chart': np.ndarray (RGB) of top-k probability chart
                - 'bbox': [x1, y1, x2, y2]
                - 'localization_method': str
                - 'status': one of 'diagnosed', 'low_confidence', 'untrained'
                - 'weights_loaded': bool
                - 'weights_load_error': Optional[str]
                - 'crop': str (e.g., 'Tomato')
                - 'condition': str (e.g., 'Late Blight')
                - 'display_name': str
                - 'confidence': float (0.0 - 1.0) — the raw model confidence,
                  regardless of status. ALWAYS check 'status' before treating
                  this (or 'crop'/'condition'/'is_healthy') as meaningful.
                - 'is_healthy': Optional[bool] — None unless status == 'diagnosed'
                - 'advice': str — agronomic advice only when status == 'diagnosed';
                  a safety explanation otherwise
                - 'confidence_threshold': the threshold that was applied (or None)
                - 'top_k': List of top predictions
        """
        # Step 1: Stage 1 - Leaf Localization
        det_result = self.detector.detect(image_input)
        bbox = det_result["bbox"]
        crop_rgb = det_result["crop_rgb"]
        image_rgb = det_result["image_rgb"]
        loc_method = det_result["method"]

        # Step 2: Stage 2 - DenseNet169 Disease Classification on ROI crop
        cls_result = self.classifier.predict(crop_rgb, top_k=top_k)
        weights_loaded = cls_result["weights_loaded"]

        # Step 2b: Determine status. This is the single decision point that
        # controls whether anything downstream is allowed to look like a
        # confident diagnosis.
        if not weights_loaded:
            status = "untrained"
        elif confidence_threshold is not None and cls_result["confidence"] < confidence_threshold:
            status = "low_confidence"
        else:
            status = "diagnosed"

        if status == "diagnosed":
            is_healthy = cls_result["is_healthy"]
            advice = cls_result["advice"]
            overlay_status = (
                Visualizer.STATUS_DIAGNOSED_HEALTHY if is_healthy else Visualizer.STATUS_DIAGNOSED_PATHOLOGY
            )
            chart_neutral = False
        else:
            # Never surface a true/false health verdict or agronomic advice
            # for an untrained or below-threshold result, no matter what the
            # raw softmax output happened to produce.
            is_healthy = None
            advice = _UNTRAINED_ADVICE if status == "untrained" else _LOW_CONFIDENCE_ADVICE
            overlay_status = (
                Visualizer.STATUS_UNTRAINED if status == "untrained" else Visualizer.STATUS_LOW_CONFIDENCE
            )
            chart_neutral = True

        # Step 3: Visual Overlay Rendering
        annotated_rgb = self.visualizer.draw_overlay(
            image_rgb=image_rgb,
            bbox=bbox,
            label=cls_result["display_name"],
            confidence=cls_result["confidence"],
            status=overlay_status,
            method=loc_method
        )

        chart_rgb = self.visualizer.render_top_k_chart(cls_result["top_k"], neutral=chart_neutral)

        return {
            "annotated_image": annotated_rgb,
            "crop_image": crop_rgb,
            "probability_chart": chart_rgb,
            "bbox": bbox,
            "localization_method": loc_method,
            "status": status,
            "weights_loaded": weights_loaded,
            "weights_load_error": self.classifier.weights_load_error,
            "crop": cls_result["crop"],
            "condition": cls_result["condition"],
            "display_name": cls_result["display_name"],
            "confidence": cls_result["confidence"],
            "is_healthy": is_healthy,
            "advice": advice,
            "confidence_threshold": confidence_threshold,
            "top_k": cls_result["top_k"],
        }
