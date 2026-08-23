"""
Plant Disease Diagnostic & Leaf Localization Web Application.
Built with Gradio for local execution and Hugging Face Spaces deployment.
"""

import os
from typing import Tuple, Dict, Optional
import gradio as gr
import numpy as np
import spaces
from PIL import Image

from src.pipeline import PlantDiagnosticPipeline
from src.weights import resolve_weights_path

# Default confidence floor for classification (Task 3). Chosen from the
# threshold sweep in reports/plantvillage_eval.json, not guessed.
#
# At the original 0.40 the floor caught 0 of 91 validation errors while
# costing 2 correct predictions — it did nothing. At 0.80: coverage 95.3%,
# selective accuracy 99.31% (up from 97.80%), 64 of 91 errors suppressed at
# a cost of 132 correct predictions reclassified as "not confident".
#
# Tuned on PlantVillage, which is lab-condition data where correct
# predictions cluster near 1.0. Field photographs will score lower overall,
# so this floor may abstain considerably more often out of distribution.
# Re-check against reports/field_eval.json once that exists.
DEFAULT_CONFIDENCE_FLOOR = 0.80


@spaces.GPU(duration=10)
def _zerogpu_probe() -> str:
    """
    Exists to satisfy ZeroGPU, which crashes at startup if a Space running on
    ZeroGPU hardware contains no @spaces.GPU-decorated function anywhere.

    Inference deliberately does NOT run on the GPU. DenseNet-169 on a single
    224x224 image takes well under a second on CPU, while each ZeroGPU call
    pays roughly 3 seconds of cold-start against a 3.5 minute daily free
    quota. Routing inference through the GPU would be slower in wall-clock
    terms AND cap the demo at roughly 70 visits per day. The Space's CPU
    container has no such limit.

    Never called in normal operation.
    """
    return "ok"



# Configuration & Paths
DEFAULT_WEIGHTS_PATH = os.path.join("models", "densenet169_plant_disease.pth")
DEFAULT_MAPPING_PATH = os.path.join("models", "class_mapping.json")

# Resolve the checkpoint: local file first (keeps dev loops network-free),
# then the Hugging Face Hub (see src/weights.py). This can fail to find
# anything at all — that's fine and expected before a checkpoint has been
# trained/uploaded. What must NOT happen is a silent fallback that looks
# like success; `weights_resolution.error` is surfaced in the UI banner
# below whenever it's set.
weights_resolution = resolve_weights_path(local_path=DEFAULT_WEIGHTS_PATH)

# Initialize Pipeline globally
pipeline = PlantDiagnosticPipeline(
    classifier_weights_path=weights_resolution.path,
    class_mapping_path=DEFAULT_MAPPING_PATH,
    detector_confidence=0.35,
)


def _supported_classes_markdown() -> str:
    """
    Builds the "Supported Crop Species & Pathologies" list from whatever is
    actually loaded in models/class_mapping.json, instead of a hand-written
    list that can silently drift out of sync with the real model head.
    """
    crop_to_conditions: Dict[str, list] = {}
    for raw_class in pipeline.classifier.idx_to_class.values():
        parsed = pipeline.classifier.parse_class_name(raw_class)
        crop_to_conditions.setdefault(parsed["crop"], []).append(parsed["condition"])

    lines = []
    for crop in sorted(crop_to_conditions):
        conditions = sorted(set(crop_to_conditions[crop]))
        lines.append(f"- **{crop}:** {', '.join(conditions)}")
    return "\n".join(lines)


def _startup_banner_markdown() -> str:
    """
    Persistent, unmissable banner shown at the top of the app whenever this
    running instance does not have a real trained checkpoint loaded. This is
    independent of the per-inference status banner in diagnose_single_image —
    it fires the moment the app starts, before anyone even uploads an image.
    """
    if pipeline.classifier.weights_loaded:
        return ""
    reason = weights_resolution.error or pipeline.classifier.weights_load_error or "Unknown reason."
    return f"""
<div style="background-color: #451a03; border: 2px solid #f59e0b; border-radius: 10px; padding: 14px 18px; margin: 0 0 20px 0;">
    <p style="margin: 0; color: #fde68a; font-weight: 700; font-size: 1.05rem;">
        ⚠ No trained checkpoint is loaded — this instance is running an UNTRAINED model.
    </p>
    <p style="margin: 6px 0 0 0; color: #fef3c7; font-size: 0.9rem;">
        Every diagnosis below will be replaced with a plain warning instead of a result.
        Reason: {reason}
    </p>
</div>
"""


def diagnose_single_image(
    image: Optional[Image.Image],
    conf_threshold: float,
    top_k_count: int,
    confidence_floor: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, float]]:
    """
    Runs two-stage diagnostic inference for a single leaf image.
    """
    if image is None:
        return None, None, None, "### ⚠️ Please upload or capture an image to run diagnosis.", {}

    pipeline.detector.confidence_threshold = conf_threshold
    result = pipeline.run(image, top_k=int(top_k_count), confidence_threshold=float(confidence_floor))

    top_k_dict = {item["display_name"]: item["probability"] for item in result["top_k"]}

    if result["status"] == "untrained":
        report_md = f"""
### ⚠️ UNTRAINED MODEL — NOT A DIAGNOSIS
<div style="background-color: #451a03; padding: 16px; border-radius: 12px; border-left: 6px solid #f59e0b; margin-bottom: 15px;">
    <p style="margin: 0; color: #fde68a; font-weight: 700; font-size: 1.05rem;">
        No trained checkpoint is loaded. The classification head is randomly initialized.
    </p>
    <p style="margin: 6px 0 0 0; color: #fef3c7; font-size: 0.9rem;">
        {result['advice']}
    </p>
</div>

---
<small style="color: #64748b;">
<b>Localization Engine:</b> {result['localization_method'].upper()} &nbsp;|&nbsp;
<b>Leaf ROI Coordinates:</b> {result['bbox']}
</small>
"""
        return (
            result["annotated_image"],
            result["crop_image"],
            result["probability_chart"],
            report_md,
            top_k_dict,
        )

    if result["status"] == "low_confidence":
        report_md = f"""
### 🟡 Low Confidence — No Diagnosis
<div style="background-color: #1e293b; padding: 16px; border-radius: 12px; border-left: 6px solid #f1c40f; margin-bottom: 15px;">
    <p style="margin: 0; color: #f8fafc; font-size: 1.0rem;">
        Best guess was <b>{result['display_name']}</b> at
        <span style="color: #f1c40f; font-weight: bold;">{result['confidence'] * 100:.1f}%</span>,
        below the confidence floor of {result['confidence_threshold'] * 100:.0f}%.
    </p>
    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.9rem;">
        {result['advice']}
    </p>
</div>

---
<small style="color: #64748b;">
<b>Localization Engine:</b> {result['localization_method'].upper()} &nbsp;|&nbsp;
<b>Leaf ROI Coordinates:</b> {result['bbox']}
</small>
"""
        return (
            result["annotated_image"],
            result["crop_image"],
            result["probability_chart"],
            report_md,
            top_k_dict,
        )

    # status == "diagnosed": current behaviour, unchanged.
    status_emoji = "✅" if result["is_healthy"] else "🚨"
    status_label = "HEALTHY (No Pathology Detected)" if result["is_healthy"] else "PATHOLOGY DETECTED"
    badge_color = "#10b981" if result["is_healthy"] else "#ef4444"

    report_md = f"""
### {status_emoji} Diagnostic Summary
<div style="background-color: #1e293b; padding: 16px; border-radius: 12px; border-left: 6px solid {badge_color}; margin-bottom: 15px;">
    <h3 style="margin: 0 0 8px 0; color: #f8fafc; font-size: 1.25rem;">
        {result['display_name']}
    </h3>
    <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
        <b>Crop:</b> <span style="color: #f1f5f9;">{result['crop']}</span> &nbsp;|&nbsp;
        <b>Condition:</b> <span style="color: #f1f5f9;">{result['condition']}</span> &nbsp;|&nbsp;
        <b>Confidence:</b> <span style="color: {badge_color}; font-weight: bold;">{result['confidence'] * 100:.1f}%</span>
    </p>
    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.9rem;">
        <b>Status:</b> <span style="color: {badge_color}; font-weight: 600;">{status_label}</span>
    </p>
</div>

#### 🌿 Recommended Agronomic Actions & Care
> {result['advice']}

---
<small style="color: #64748b;">
<b>Localization Engine:</b> {result['localization_method'].upper()} &nbsp;|&nbsp;
<b>Leaf ROI Coordinates:</b> {result['bbox']}
</small>
"""

    return (
        result["annotated_image"],
        result["crop_image"],
        result["probability_chart"],
        report_md,
        top_k_dict
    )


# Custom CSS for Modern, Premium Agritech Styling
custom_css = """
#app-container {
    max-width: 1280px;
    margin: 0 auto;
}
.header-badge {
    display: inline-block;
    padding: 4px 12px;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    font-weight: 600;
    border-radius: 9999px;
    font-size: 0.85rem;
    margin-bottom: 8px;
}
.main-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card-panel {
    background: #0f172a !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}
"""

with gr.Blocks(title="🌿 Plant Disease Diagnostic Pipeline | DenseNet-169") as demo:
    with gr.Column(elem_id="app-container"):

        # Header Section
        gr.HTML(f"""
        <div style="text-align: center; margin: 15px 0 25px 0;">
            <span class="header-badge">AI Plant Pathology v1.0</span>
            <h1 class="main-title">🌿 Plant Disease Detection & Visual Diagnostic Pipeline</h1>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 780px; margin: 0 auto;">
                Two-stage computer vision system integrating <b>Leaf Localization</b> with a
                fine-tuned <b>DenseNet-169</b> convolutional neural network across
                {len(pipeline.classifier.idx_to_class)} classes (see the "System Architecture" tab
                for exactly which ones — this number is read from models/class_mapping.json, not hardcoded).
            </p>
        </div>
        """)

        # Persistent, unmissable warning if this instance has no real checkpoint loaded.
        # Empty string when a checkpoint IS loaded, so this renders nothing in that case.
        gr.HTML(_startup_banner_markdown())

        with gr.Tabs():
            # TAB 1: Single Image Diagnostic
            with gr.TabItem("🔍 Real-Time Leaf Diagnosis"):
                with gr.Row():
                    # Left Column: Inputs & Controls
                    with gr.Column(scale=5):
                        input_img = gr.Image(
                            type="pil",
                            label="Upload or Capture Leaf Photo",
                            sources=["upload", "webcam", "clipboard"],
                            elem_classes=["card-panel"]
                        )

                        with gr.Accordion("⚙️ Detection & Inference Parameters", open=False):
                            conf_slider = gr.Slider(
                                minimum=0.1,
                                maximum=0.95,
                                value=0.35,
                                step=0.05,
                                label="Localization Confidence Threshold",
                                info="Stage 1 only: how confident the leaf-region detector must be."
                            )
                            confidence_floor_slider = gr.Slider(
                                minimum=0.0,
                                maximum=0.95,
                                value=DEFAULT_CONFIDENCE_FLOOR,
                                step=0.01,
                                label="Diagnosis Confidence Floor",
                                info=(
                                    "Stage 2: below this, results are shown as 'low confidence, "
                                    "not diagnostic' instead of a health verdict. Default "
                                    f"({DEFAULT_CONFIDENCE_FLOOR}) is a starting guess, not a "
                                    "calibrated value — see README."
                                )
                            )
                            top_k_slider = gr.Slider(
                                minimum=3,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Top-K Probability Display Count"
                            )

                        diagnose_btn = gr.Button("🌿 Run Diagnostic Analysis", variant="primary", size="lg")
                        clear_btn = gr.ClearButton(components=[input_img], value="Clear Input")

                        # Bundled PlantVillage samples so visitors have something
                        # to click. These are lab-condition images from the
                        # dataset the model was trained on, so they represent the
                        # easy case, not field performance. See README.
                        _sample_dir = os.path.join("data", "samples")
                        if os.path.isdir(_sample_dir):
                            _samples = sorted(
                                os.path.join(_sample_dir, f)
                                for f in os.listdir(_sample_dir)
                                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                            )
                            if _samples:
                                gr.Examples(
                                    examples=_samples,
                                    inputs=input_img,
                                    label="Sample leaves (PlantVillage, lab conditions)",
                                )

                    # Right Column: Outputs
                    with gr.Column(scale=7):
                        diagnostic_output_md = gr.Markdown("### 📋 Upload a leaf image on the left and click 'Run Diagnostic Analysis'.")

                        with gr.Row():
                            annotated_output_img = gr.Image(
                                label="🎯 Stage 1: Leaf Localization & Overlay",
                                interactive=False
                            )
                            crop_output_img = gr.Image(
                                label="🔍 Extracted Leaf ROI",
                                interactive=False
                            )

                        with gr.Row():
                            prob_chart_img = gr.Image(
                                label="📊 Diagnostic Confidence Distribution",
                                interactive=False
                            )
                            label_distribution = gr.Label(
                                label="Top Class Ranking",
                                num_top_classes=5
                            )

                # Connect Interaction
                diagnose_btn.click(
                    fn=diagnose_single_image,
                    inputs=[input_img, conf_slider, top_k_slider, confidence_floor_slider],
                    outputs=[
                        annotated_output_img,
                        crop_output_img,
                        prob_chart_img,
                        diagnostic_output_md,
                        label_distribution
                    ]
                )

            # TAB 2: Model Architecture & System Pipeline
            with gr.TabItem("🏗️ System Architecture & Model Details"):
                gr.Markdown(f"""
### 🧠 Two-Stage Computer Vision Architecture

The diagnostic framework divides pathology detection into specialized phases to maximize classification accuracy:

```
[ Input Leaf Image (RGB) ]
           │
           ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 1: Leaf Localization & Region-of-Interest Extraction│
│ • OpenCV Contour / HSV Color Space Morphology Segmentation│
│ • Optional YOLOv8 Foliage Detection Integration           │
│ • Bounding Box Coordinates: [x1, y1, x2, y2]              │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼ (ROI Crop)
┌───────────────────────────────────────────────────────────┐
│ Stage 2: DenseNet-169 Feature Extraction & Classification │
│ • 169-layer Densely Connected Convolutional Network       │
│ • Feature Reuse: concatenates feature maps from all prior │
│   layers to minimize vanishing gradients                  │
│ • Customized Dropout (p=0.3) + Linear Classification Head │
│ • Softmax Probability Output across {len(pipeline.classifier.idx_to_class)} Disease Classes    │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 3: OpenCV Visual Overlay & Diagnostic Reporting     │
│ • Bounding box + status badge (only when status=diagnosed)│
│ • Neutral badge for untrained / low-confidence results    │
│ • Agronomic advice ONLY when status == 'diagnosed'         │
└───────────────────────────────────────────────────────────┘
```

#### 🌿 Supported Crop Species & Pathologies
This list is generated from `models/class_mapping.json` as currently loaded by this
running instance — it will not silently drift out of sync with the actual model head.

""" + _supported_classes_markdown())

            # TAB 3: About & Instructions
            with gr.TabItem("ℹ️ Setup & Training Guidelines"):
                gr.Markdown("""
### 🚀 Local Execution & Model Training

1. **Train / Fine-Tune Model:**
   - Two notebooks exist under `notebooks/`. Check the README's "Model training"
     section for which one currently contains a complete, working training loop
     before relying on either — they are not equivalent, and one is a skeleton
     that does not actually train anything if run top to bottom.
   - Save the trained checkpoint to `models/densenet169_plant_disease.pth`
     for local dev, **or** upload it to a Hugging Face model repo and set
     `PLANT_DISEASE_HF_REPO_ID` (see README) — the app will download and
     cache it automatically at startup if no local file is present.

2. **Run Web Interface Locally:**
   ```bash
   python app.py
   ```
   If neither a local checkpoint nor `PLANT_DISEASE_HF_REPO_ID` is configured,
   the app still starts, but every result is replaced with an explicit
   "untrained model" warning instead of a diagnosis — see the banner at the
   top of this page.

3. **Deploy to Hugging Face Spaces:**
   - Push this repository to a Hugging Face Space with SDK set to `gradio`.
   - Set `PLANT_DISEASE_HF_REPO_ID` (and `PLANT_DISEASE_HF_FILENAME` if you
     didn't use the default filename) as a Space secret/variable so the
     Space can fetch the checkpoint without it living in git.
                """)

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 12px; color: #64748b; font-size: 0.85rem; border-top: 1px solid #1e293b;">
            Plant Disease DenseNet-169 Diagnostic Pipeline • Open-Source Research & Educational Application
        </div>
        """)

if __name__ == "__main__":
    demo.launch(css=custom_css, share=False)
