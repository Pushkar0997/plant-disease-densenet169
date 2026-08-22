"""
OpenCV & Matplotlib Visual Overlay and Diagnostic Annotation Engine.
Renders stylized bounding boxes, status badges, and probability charts.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image


class Visualizer:
    """
    Renders high-contrast bounding box overlays, diagnostic labels,
    and prediction charts onto images.
    """

    # Color definitions (BGR for OpenCV)
    COLOR_HEALTHY_BGR = (46, 204, 113)     # Emerald Green
    COLOR_DISEASED_BGR = (39, 60, 235)     # Vibrant Crimson/Red-Orange
    COLOR_BADGE_BG_BGR = (25, 25, 25)      # Dark Charcoal
    COLOR_WHITE_BGR = (255, 255, 255)
    COLOR_ACCENT_BGR = (241, 196, 15)      # Gold

    def __init__(self, font_scale: float = 0.65, thickness: int = 2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_overlay(
        self,
        image_rgb: np.ndarray,
        bbox: List[int],
        label: str,
        confidence: float,
        is_healthy: bool,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws an aesthetic bounding box and diagnostic badge on the RGB image.

        Args:
            image_rgb: Input image in RGB format.
            bbox: [x1, y1, x2, y2]
            label: Diagnostic text label to display.
            confidence: Classification or detection confidence (0.0 - 1.0).
            is_healthy: True if plant is diagnosed healthy, False if diseased.
            method: Localization method ('yolov8' / 'contour_segmentation').

        Returns:
            Annotated RGB numpy image.
        """
        # Convert RGB to BGR for OpenCV rendering
        annotated = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
        h, w, _ = annotated.shape
        x1, y1, x2, y2 = bbox

        # Select color based on health status
        theme_color = self.COLOR_HEALTHY_BGR if is_healthy else self.COLOR_DISEASED_BGR

        # 1. Draw main bounding box with rounded aesthetic or sleek corner markers
        cv2.rectangle(annotated, (x1, y1), (x2, y2), theme_color, self.thickness, cv2.LINE_AA)

        # Corner bracket accents for modern HUD look
        corner_len = min(25, max(10, int(min(x2 - x1, y2 - y1) * 0.15)))
        corner_thick = self.thickness + 2

        # Top-Left
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), theme_color, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), theme_color, corner_thick, cv2.LINE_AA)
        # Top-Right
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), theme_color, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), theme_color, corner_thick, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), theme_color, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), theme_color, corner_thick, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), theme_color, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), theme_color, corner_thick, cv2.LINE_AA)

        # 2. Prepare badge text
        status_text = "HEALTHY" if is_healthy else "PATHOLOGY DETECTED"
        badge_text_main = f"{label} ({confidence * 100:.1f}%)"
        badge_text_sub = f"Status: {status_text}"

        (w1, h1), b1 = cv2.getTextSize(badge_text_main, self.font, self.font_scale, 1)
        (w2, h2), b2 = cv2.getTextSize(badge_text_sub, self.font, self.font_scale * 0.8, 1)

        badge_w = max(w1, w2) + 20
        badge_h = h1 + h2 + 25

        # Place badge above bbox if space permits, otherwise inside
        badge_y1 = max(10, y1 - badge_h - 8)
        if badge_y1 <= 15:
            badge_y1 = y1 + 10
        badge_y2 = badge_y1 + badge_h
        badge_x1 = max(10, min(x1, w - badge_w - 10))
        badge_x2 = badge_x1 + badge_w

        # Draw semi-transparent background badge
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (badge_x1, badge_y1),
            (badge_x2, badge_y2),
            self.COLOR_BADGE_BG_BGR,
            -1
        )
        # Left accent pill bar
        cv2.rectangle(
            overlay,
            (badge_x1, badge_y1),
            (badge_x1 + 5, badge_y2),
            theme_color,
            -1
        )

        alpha = 0.85
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Draw Badge Text
        cv2.putText(
            annotated,
            badge_text_main,
            (badge_x1 + 12, badge_y1 + h1 + 8),
            self.font,
            self.font_scale,
            self.COLOR_WHITE_BGR,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            badge_text_sub,
            (badge_x1 + 12, badge_y1 + h1 + h2 + 16),
            self.font,
            self.font_scale * 0.8,
            theme_color,
            1,
            cv2.LINE_AA
        )

        # Convert back to RGB for return
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def render_top_k_chart(self, top_k: List[Dict[str, Any]]) -> np.ndarray:
        """
        Creates a clean horizontal bar chart visualization of top-k class probabilities.

        Args:
            top_k: List of prediction dictionaries with 'display_name' and 'probability'.

        Returns:
            RGB numpy array image of the rendered chart.
        """
        names = [item["display_name"] for item in reversed(top_k)]
        probs = [item["probability"] * 100 for item in reversed(top_k)]
        colors = ["#2ecc71" if item["is_healthy"] else "#e74c3c" for item in reversed(top_k)]

        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#0f172a")

        bars = ax.barh(names, probs, color=colors, height=0.55, edgecolor="none")

        ax.set_xlim(0, 105)
        ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=10, fontweight="bold")
        ax.set_title("Top Class Diagnostic Probabilities", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)

        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.6)

        # Add percentage labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                color="#f1f5f9",
                fontsize=9,
                fontweight="bold"
            )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        chart_pil = Image.open(buf).convert("RGB")
        return np.array(chart_pil)
