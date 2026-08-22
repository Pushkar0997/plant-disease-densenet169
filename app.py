"""
Plant Disease Diagnostic & Leaf Localization Web Application.
Built with Gradio for local execution and Hugging Face Spaces deployment.
"""

import os
from typing import Tuple, Dict, Any, List
import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline import PlantDiagnosticPipeline


# Configuration & Paths
DEFAULT_WEIGHTS_PATH = os.path.join("models", "densenet169_plant_disease.pth")
DEFAULT_MAPPING_PATH = os.path.join("models", "class_mapping.json")

# Initialize Pipeline globally
pipeline = PlantDiagnosticPipeline(
    classifier_weights_path=DEFAULT_WEIGHTS_PATH,
    class_mapping_path=DEFAULT_MAPPING_PATH,
    detector_confidence=0.35,
)


def diagnose_single_image(
    image: Image.Image,
    conf_threshold: float,
    top_k_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, float]]:
    """
    Runs two-stage diagnostic inference for a single leaf image.
    """
    if image is None:
        return None, None, None, "### ⚠️ Please upload or capture an image to run diagnosis.", {}

    pipeline.detector.confidence_threshold = conf_threshold
    result = pipeline.run(image, top_k=int(top_k_count))

    # Build markdown diagnosis report
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

    # Build dictionary for Gradio Label output
    top_k_dict = {item["display_name"]: item["probability"] for item in result["top_k"]}

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

with gr.Blocks(css=custom_css, title="🌿 Plant Disease Diagnostic Pipeline | DenseNet-169") as demo:
    with gr.Column(elem_id="app-container"):
        
        # Header Section
        gr.HTML("""
        <div style="text-align: center; margin: 15px 0 25px 0;">
            <span class="header-badge">AI Plant Pathology v1.0</span>
            <h1 class="main-title">🌿 Plant Disease Detection & Visual Diagnostic Pipeline</h1>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 780px; margin: 0 auto;">
                Two-stage deep learning diagnostic system integrating <b>Leaf Localization</b> with a 
                fine-tuned <b>DenseNet-169</b> convolutional neural network for 38 crop disease pathologies.
            </p>
        </div>
        """)

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
                                label="Localization Confidence Threshold"
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
                    inputs=[input_img, conf_slider, top_k_slider],
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
                gr.Markdown("""
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
│ • Softmax Probability Output across 38 Disease Classes    │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 3: OpenCV Visual Overlay & Diagnostic Reporting     │
│ • Real-time HUD Bounding Box Rendering                    │
│ • Status Tagging (Healthy vs Pathological)                │
│ • Agronomic Action Advice Generation                      │
└───────────────────────────────────────────────────────────┘
```

#### 🌿 Supported Crop Species & Pathologies (PlantVillage 38 Classes)
- **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry:** Healthy
- **Cherry:** Powdery Mildew, Healthy
- **Corn (Maize):** Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape:** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy
- **Orange:** Citrus Greening (Huanglongbing)
- **Peach:** Bacterial Spot, Healthy
- **Pepper (Bell):** Bacterial Spot, Healthy
- **Potato:** Early Blight, Late Blight, Healthy
- **Raspberry & Soybean:** Healthy
- **Squash:** Powdery Mildew
- **Strawberry:** Leaf Scorch, Healthy
- **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
                """)

            # TAB 3: About & Instructions
            with gr.TabItem("ℹ️ Setup & Training Guidelines"):
                gr.Markdown("""
### 🚀 Local Execution & Model Training

1. **Train / Fine-Tune Model:**
   - Explore and run [`notebooks/01_densenet169_training_pipeline.ipynb`](file:///d:/Coding_Work/plant-disease-densenet169/notebooks/01_densenet169_training_pipeline.ipynb) on Google Colab or locally with GPU.
   - Save trained checkpoint to `models/densenet169_plant_disease.pth`.

2. **Run Web Interface Locally:**
   ```bash
   python app.py
   ```

3. **Deploy to Hugging Face Spaces:**
   - Push this repository to a Hugging Face Space with SDK set to `gradio`.
                """)

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 12px; color: #64748b; font-size: 0.85rem; border-top: 1px solid #1e293b;">
            Plant Disease DenseNet-169 Diagnostic Pipeline • Open-Source Research & Educational Application
        </div>
        """)

if __name__ == "__main__":
    demo.launch(share=False)
