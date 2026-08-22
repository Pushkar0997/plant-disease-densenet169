"""
Dry Run Verification Script for Plant Disease Diagnostic Pipeline.
Runs inference on sample leaf images and verifies each stage of the pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image
from src.pipeline import PlantDiagnosticPipeline


def run_dry_test(image_path: Optional[str] = None, save_output: bool = True):
    print("=" * 65)
    print("[*] Plant Disease DenseNet-169 Pipeline - Dry Run Test")
    print("=" * 65)

    # 1. Check/Select sample image
    samples_dir = Path("data/samples")
    if image_path is None:
        extensions = ("*.JPG", "*.jpg", "*.png", "*.jpeg")
        sample_files = []
        for ext in extensions:
            sample_files.extend(samples_dir.glob(ext))

        # Filter out generated test outputs
        sample_files = [f for f in sample_files if not f.name.startswith("dry_run_")]
        if not sample_files:
            print(f"[!] No sample images found in {samples_dir}. Please supply an image path.")
            return False
        image_path = str(sample_files[0])

    print(f"\n[1/4] Loading input image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return False

    image = Image.open(image_path)
    print(f"      Image format: {image.format}, Size: {image.size}, Mode: {image.mode}")

    # 2. Initialize Pipeline
    print("\n[2/4] Initializing Two-Stage Diagnostic Pipeline...")
    weights_path = os.path.join("models", "densenet169_plant_disease.pth")
    mapping_path = os.path.join("models", "class_mapping.json")

    pipeline = PlantDiagnosticPipeline(
        classifier_weights_path=weights_path if os.path.exists(weights_path) else None,
        class_mapping_path=mapping_path,
        detector_confidence=0.35
    )

    # 3. Run Inference
    print("\n[3/4] Running Diagnostic Inference (Localization -> Classification -> Overlay)...")
    result = pipeline.run(image, top_k=5)

    # 4. Display Results
    print("\n[4/4] Diagnostic Summary Results:")
    print("-" * 55)
    print(f"  * Diagnosed Crop:       {result['crop']}")
    print(f"  * Pathology Condition:  {result['condition']}")
    print(f"  * Full Label:           {result['display_name']}")
    print(f"  * Confidence Score:     {result['confidence'] * 100:.2f}%")
    print(f"  * Health Status:        {'HEALTHY' if result['is_healthy'] else 'PATHOLOGY DETECTED'}")
    print(f"  * Leaf Bounding Box:    {result['bbox']}")
    print(f"  * Localization Method:  {result['localization_method']}")
    print(f"  * Agronomic Advice:     {result['advice']}")
    print("-" * 55)
    print("  * Top-5 Class Distributions:")
    for rank, item in enumerate(result['top_k'], 1):
        print(f"    {rank}. {item['display_name']:<35} -> {item['probability'] * 100:.2f}%")
    print("-" * 55)

    if save_output:
        out_dir = Path("data/samples")
        out_path = out_dir / "dry_run_annotated_output.jpg"
        out_chart = out_dir / "dry_run_probability_chart.png"

        annotated_img = Image.fromarray(result["annotated_image"])
        annotated_img.save(out_path)

        chart_img = Image.fromarray(result["probability_chart"])
        chart_img.save(out_chart)

        print(f"\n[+] Saved annotated result to: {out_path}")
        print(f"[+] Saved probability chart to: {out_chart}")

    print("\n[SUCCESS] Dry run completed successfully! All pipeline stages functional.\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry run plant disease diagnosis")
    parser.add_argument("--image", type=str, default=None, help="Path to input leaf image")
    args = parser.parse_args()

    success = run_dry_test(args.image)
    if not success:
        sys.exit(1)
