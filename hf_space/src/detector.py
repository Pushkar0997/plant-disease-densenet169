"""
Leaf Localization and Region of Interest (ROI) Detection Module.
Provides YOLOv8 deep learning detection with automatic fallback to OpenCV contour & color segmentation.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image


class LeafDetector:
    """
    Detects and localizes plant leaves within input imagery to extract
    high-quality Regions of Interest (ROI) for downstream classification.
    """

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        padding_ratio: float = 0.05,
    ):
        """
        Initialize the LeafDetector.

        Args:
            yolo_model_path: Path to YOLOv8 weights (e.g., 'yolov8n.pt' or custom). If None, uses contour segmentation.
            confidence_threshold: Minimum confidence score for detection.
            padding_ratio: Relative margin added around detected bounding box.
        """
        self.confidence_threshold = confidence_threshold
        self.padding_ratio = padding_ratio
        self.yolo_model = None
        self.use_yolo = False

        if yolo_model_path:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_model_path)
                self.use_yolo = True
            except Exception as e:
                print(f"[LeafDetector] Warning: Could not initialize YOLO model from {yolo_model_path}: {e}")
                print("[LeafDetector] Falling back to OpenCV contour/color segmentation.")
                self.use_yolo = False

    def _to_numpy_rgb(self, image: Any) -> np.ndarray:
        """Converts input PIL Image, path, or array into RGB numpy array."""
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"Could not load image from path: {image}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _contour_segmentation_detection(self, img_rgb: np.ndarray) -> Tuple[list, float]:
        """
        Extracts leaf bounding box using HSV color masking and contour analysis.
        """
        h, w, _ = img_rgb.shape
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Plant leaf green-yellow-brown hue ranges in HSV
        lower_green = np.array([20, 30, 20])
        upper_green = np.array([95, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_green, upper_green)

        # Foliage / brown rust lesions range
        lower_brown = np.array([10, 40, 20])
        upper_brown = np.array([25, 255, 200])
        mask2 = cv2.inRange(img_hsv, lower_brown, upper_brown)

        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological filtering to remove noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: Otsu thresholding on grayscale channel
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Full image bounding box fallback
            return [0, 0, w, h], 0.50

        # Find largest contour by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > (0.01 * w * h)]
        if not valid_contours:
            largest_c = max(contours, key=cv2.contourArea)
        else:
            largest_c = max(valid_contours, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(largest_c)

        # Apply padding
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        # Confidence estimated by salient area ratio
        area_ratio = (bw * bh) / float(w * h)
        confidence = float(np.clip(0.65 + (area_ratio * 0.3), 0.50, 0.95))

        return [int(x1), int(y1), int(x2), int(y2)], confidence

    def detect(self, image: Any) -> Dict[str, Any]:
        """
        Localize leaf region in the given image.

        Args:
            image: PIL Image, filepath, or numpy ndarray (RGB).

        Returns:
            Dict containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'method': 'yolov8' or 'contour_segmentation'
                - 'crop_rgb': np.ndarray of the cropped leaf ROI
                - 'image_rgb': original image as np.ndarray
        """
        img_rgb = self._to_numpy_rgb(image)
        h, w, _ = img_rgb.shape

        bbox = [0, 0, w, h]
        confidence = 0.50
        method = "contour_segmentation"

        if self.use_yolo and self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=img_rgb,
                    conf=self.confidence_threshold,
                    verbose=False
                )
                if len(results) > 0 and len(results[0].boxes) > 0:
                    best_box = results[0].boxes[0]
                    coords = best_box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    bbox = [int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))]
                    confidence = float(best_box.conf[0].cpu().item())
                    method = "yolov8"
                else:
                    bbox, confidence = self._contour_segmentation_detection(img_rgb)
            except Exception as e:
                print(f"[LeafDetector] YOLO detection failed: {e}. Using contour segmentation.")
                bbox, confidence = self._contour_segmentation_detection(img_rgb)
        else:
            bbox, confidence = self._contour_segmentation_detection(img_rgb)

        x1, y1, x2, y2 = bbox
        # Ensure non-zero crop
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h
            bbox = [x1, y1, x2, y2]

        crop_rgb = img_rgb[y1:y2, x1:x2]

        return {
            "bbox": bbox,
            "confidence": confidence,
            "method": method,
            "crop_rgb": crop_rgb,
            "image_rgb": img_rgb,
        }
