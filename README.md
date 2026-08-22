# 🌿 Plant Disease Detection & Visual Diagnostic Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Backbone-DenseNet--169-brightgreen.svg)](https://pytorch.org/vision/main/models/densenet.html)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision pipeline combining **leaf localization** with fine-tuned **DenseNet169** transfer learning to diagnose crop leaf pathologies and generate real-time bounding box visual overlays.

---

## 📌 Project Overview

This project provides a modern, modular computer vision pipeline for agricultural disease diagnosis. 

Standard deep learning classifiers often suffer when background noise, multiple plants, or varying distances are present in unconstrained field photographs. This pipeline addresses that challenge with a **two-stage diagnostic architecture**:

1. **Localization Stage:** Detects and extracts the primary leaf region within an image using contour/color morphology segmentation or YOLOv8.
2. **Classification Stage:** Processes the cropped Region of Interest (ROI) through a fine-tuned **DenseNet-169** deep convolutional network to classify disease states across 38 distinct crop-pathology categories.
3. **Visual Overlay Engine:** Uses OpenCV to render dynamic HUD bounding box overlays, diagnostic status badges, and confidence metrics directly onto the original image.
4. **Agronomic Recommendations:** Provides actionable treatment tips and management strategies based on the identified disease condition.

---

## 🏗️ System Architecture

```
[ Input Image (RGB) ]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 1: Leaf Localization Engine                         │
│ • OpenCV Contour / HSV Color Space Morphology             │
│ • Optional YOLOv8 Foliage Detection Integration           │
│ ──► Generates Bounding Box Coordinates: [x1, y1, x2, y2]  │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼ (ROI Crop)
┌───────────────────────────────────────────────────────────┐
│ Stage 2: DenseNet-169 Feature Extractor & Classifier      │
│ • 169-layer Densely Connected Convolutional Network       │
│ • Custom Dropout (p=0.3) + Linear Head                    │
│ ──► Outputs Softmax Logits & Top-K Class Probabilities    │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 3: Visual Overlay & Diagnostic Summary Engine       │
│ • OpenCV Bounding Box + Status Badge (Healthy/Pathology)  │
│ • Matplotlib Horizontal Probability Distribution Chart    │
│ • Agronomic Action Advice Generation                      │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
[ Annotated Image + Diagnostic Report + Probability Chart ]
```

---

## 📂 Directory Structure

```plaintext
plant-disease-densenet169/
├── .github/
│   └── workflows/
│       └── python-app.yml       # CI workflow for linting & syntax verification
├── assets/                      # Demo images, architecture diagrams for README
├── data/                        # Local testing samples (git-ignored)
│   ├── raw/
│   └── samples/
├── models/                      # Checkpoints & mappings (weights git-ignored)
│   ├── class_mapping.json       # Exported index-to-class dictionary (38 classes)
│   └── .gitkeep
├── notebooks/                   # Google Colab / Jupyter exploration
│   └── 01_densenet169_training_pipeline.ipynb
├── src/                         # Modular Python package
│   ├── __init__.py
│   ├── detector.py              # Leaf localization logic (YOLOv8 / Contours)
│   ├── classifier.py            # DenseNet169 model definition & loading
│   ├── visualizer.py            # OpenCV bounding box & label overlay rendering
│   └── pipeline.py              # Two-stage inference pipeline coordinator
├── app.py                       # Gradio Web UI (Hugging Face Spaces entrypoint)
├── .gitignore                   # Ignores data, checkpoints, venv, cache
├── LICENSE                      # MIT License
├── README.md                    # Project documentation & architectural breakdown
└── requirements.txt             # Project dependencies
```

---

## 🌿 Supported Classes (PlantVillage 38 Classes)

| Crop | Supported Conditions / Pathologies |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Citrus Greening (Huanglongbing) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/plant-disease-densenet169.git
cd plant-disease-densenet169

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Gradio Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860` to access the interactive diagnostic interface.

### 3. Programmatic Usage in Python
```python
from src.pipeline import PlantDiagnosticPipeline
from PIL import Image

# Initialize pipeline
pipeline = PlantDiagnosticPipeline(
    classifier_weights_path="models/densenet169_plant_disease.pth",
    class_mapping_path="models/class_mapping.json"
)

# Run diagnostic inference
image = Image.open("data/samples/sample_leaf.jpg")
result = pipeline.run(image, top_k=5)

print(f"Diagnosed Condition: {result['display_name']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
print(f"Status: {'Healthy' if result['is_healthy'] else 'Pathology Detected'}")
print(f"Recommended Action: {result['advice']}")

# Save or display annotated image
annotated_img = Image.fromarray(result["annotated_image"])
annotated_img.save("data/samples/annotated_result.jpg")
```

---

## 🔬 Model Training & Fine-Tuning

The full training pipeline is provided in [`notebooks/01_densenet169_training_pipeline.ipynb`](file:///d:/Coding_Work/plant-disease-densenet169/notebooks/01_densenet169_training_pipeline.ipynb).

### Training Highlights:
- **Architecture:** DenseNet-169 with custom Dropout (0.3) + Linear classification head.
- **Augmentation:** Random resized crop, horizontal/vertical flips, affine rotations, and color jittering.
- **Optimization:** AdamW optimizer (`lr=3e-4`, `weight_decay=1e-4`) with Cosine Annealing learning rate schedule.
- **Mixed Precision:** Automatic Mixed Precision (`torch.cuda.amp`) for accelerated GPU computation.
- **Regularization:** Cross-entropy loss with label smoothing (`0.1`).

---

## 🚢 Deployment to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) selecting **Gradio** as the SDK.
2. Push this repository:
```bash
git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
git push space main
```
The `app.py` and `requirements.txt` in the root directory will automatically build and launch the application.

---

## 📜 License

This project is open-source under the [MIT License](file:///d:/Coding_Work/plant-disease-densenet169/LICENSE).
