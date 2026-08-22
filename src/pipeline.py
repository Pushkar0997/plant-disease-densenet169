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
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute full two-stage diagnostic workflow on an input image.

        Args:
            image_input: Filepath, PIL Image, or RGB numpy array.
            top_k: Number of highest ranking classes to compute.

        Returns:
            Dictionary containing:
                - 'annotated_image': np.ndarray (RGB) with bounding box and badges
                - 'crop_image': np.ndarray (RGB) of localized leaf ROI
                - 'probability_chart': np.ndarray (RGB) of top-k probability chart
                - 'bbox': [x1, y1, x2, y2]
                - 'localization_method': str
                - 'crop': str (e.g., 'Tomato')
                - 'condition': str (e.g., 'Late Blight')
                - 'display_name': str
                - 'confidence': float (0.0 - 1.0)
                - 'is_healthy': bool
                - 'advice': str (Agronomic / treatment recommendation)
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

        # Step 3: Visual Overlay Rendering
        annotated_rgb = self.visualizer.draw_overlay(
            image_rgb=image_rgb,
            bbox=bbox,
            label=cls_result["display_name"],
            confidence=cls_result["confidence"],
            is_healthy=cls_result["is_healthy"],
            method=loc_method
        )

        chart_rgb = self.visualizer.render_top_k_chart(cls_result["top_k"])

        return {
            "annotated_image": annotated_rgb,
            "crop_image": crop_rgb,
            "probability_chart": chart_rgb,
            "bbox": bbox,
            "localization_method": loc_method,
            "crop": cls_result["crop"],
            "condition": cls_result["condition"],
            "display_name": cls_result["display_name"],
            "confidence": cls_result["confidence"],
            "is_healthy": cls_result["is_healthy"],
            "advice": cls_result["advice"],
            "top_k": cls_result["top_k"],
        }
