"""
Evaluate the full two-stage pipeline on your own field photographs.

Unlike scripts/evaluate.py, this runs localisation AND classification — the
whole thing the pipeline actually is — because the two-stage design exists
specifically to handle messy photos, and evaluating only the classifier would
skip the stage that is most likely to fail.

Why this script exists
----------------------
PlantVillage is lab-condition imagery: one leaf, uniform background,
controlled lighting. Accuracy on it says nothing about a phone photo taken in
a field with soil, sky, multiple overlapping leaves and hard shadows in frame.
Those are different distributions and the gap is usually large.

On sample size
--------------
You mentioned 20-30 photos. That is enough to notice a catastrophic failure
and nothing more. At n=25, a 68% accuracy has a 95% confidence interval of
roughly 48%-83% — the interval is wider than most differences you would care
about. This script prints a Wilson confidence interval next to every rate for
exactly that reason, and refuses to present bare percentages that would imply
more precision than 25 photos can support. Treat the output as a smoke test.

Labels CSV
----------
Two columns, header required:

    filename,label
    IMG_0001.jpg,Tomato_Late_blight
    IMG_0002.jpg,Pepper__bell___healthy

`label` must exactly match a class name in class_mapping.json. The script
lists any unrecognised labels and stops rather than scoring against names the
model has no concept of.

Usage
-----
    python scripts/evaluate_field.py \
        --images-dir data/field \
        --labels-csv data/field/labels.csv \
        --checkpoint /content/densenet169_plantvillage.pth \
        --output reports/field_eval.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import PlantDiagnosticPipeline  # noqa: E402
from src.weights import resolve_weights_path  # noqa: E402


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """
    Wilson score interval. Used instead of the normal approximation because
    at n=25 the normal approximation is unreliable and can produce bounds
    outside [0,1], which would misrepresent how little these photos establish.
    """
    if n == 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def rate_block(successes: int, n: int) -> Dict:
    lo, hi = wilson_interval(successes, n)
    return {
        "count": int(successes),
        "of": int(n),
        "rate": (successes / n) if n else None,
        "wilson_95_ci": [lo, hi],
    }


def summarize(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--labels-csv", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path("models/densenet169_plant_disease.pth"))
    p.add_argument("--class-mapping", type=Path, default=Path("models/class_mapping.json"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--confidence-threshold", type=float, default=0.40,
                   help="Diagnosis confidence floor, matching the app default")
    p.add_argument("--full-frame-ratio", type=float, default=0.90,
                   help="bbox area / image area at or above which localisation is "
                        "counted as having failed (it just returned the whole frame)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    resolution = resolve_weights_path(local_path=str(args.checkpoint))
    if resolution.error:
        print(f"[field] checkpoint resolution: {resolution.error}")

    pipeline = PlantDiagnosticPipeline(
        classifier_weights_path=resolution.path,
        class_mapping_path=str(args.class_mapping),
        device=args.device,
    )
    if not pipeline.classifier.weights_loaded:
        print(f"\nERROR: no trained checkpoint loaded: "
              f"{pipeline.classifier.weights_load_error}", file=sys.stderr)
        print("Refusing to report metrics for an untrained model.", file=sys.stderr)
        return 1

    known = set(pipeline.classifier.idx_to_class.values())

    with open(args.labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: no rows in {args.labels_csv}", file=sys.stderr)
        return 1
    for col in ("filename", "label"):
        if col not in rows[0]:
            print(f"ERROR: {args.labels_csv} needs a '{col}' column.", file=sys.stderr)
            return 1

    unknown = sorted({r["label"] for r in rows} - known)
    if unknown:
        print("ERROR: labels not present in the class mapping:", file=sys.stderr)
        for label in unknown:
            print(f"  {label}", file=sys.stderr)
        print(f"\nValid class names ({len(known)}):", file=sys.stderr)
        for name in sorted(known):
            print(f"  {name}", file=sys.stderr)
        return 1

    per_image = []
    correct_conf, incorrect_conf = [], []
    n_correct = n_full_frame = n_below_threshold = 0

    for i, row in enumerate(rows, 1):
        path = args.images_dir / row["filename"]
        if not path.exists():
            print(f"  [{i}/{len(rows)}] MISSING {path}, skipping")
            continue

        image = Image.open(path).convert("RGB")
        result = pipeline.run(image, top_k=5, confidence_threshold=args.confidence_threshold)

        x1, y1, x2, y2 = result["bbox"]
        w, h = image.size
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h) if w and h else 0.0
        full_frame = area_ratio >= args.full_frame_ratio

        is_correct = result["top_k"][0]["raw_name"] == row["label"]
        confidence = result["confidence"]

        n_correct += int(is_correct)
        n_full_frame += int(full_frame)
        n_below_threshold += int(result["status"] == "low_confidence")
        (correct_conf if is_correct else incorrect_conf).append(confidence)

        per_image.append({
            "filename": row["filename"],
            "true_label": row["label"],
            "predicted_label": result["top_k"][0]["raw_name"],
            "correct": bool(is_correct),
            "confidence": float(confidence),
            "status": result["status"],
            "bbox": result["bbox"],
            "bbox_area_ratio": float(area_ratio),
            "localisation_covered_full_frame": bool(full_frame),
            "localisation_method": result["localization_method"],
        })
        flag = "OK " if is_correct else "MISS"
        print(f"  [{i}/{len(rows)}] {flag} {row['filename']} "
              f"-> {result['top_k'][0]['raw_name']} ({confidence:.3f}, {result['status']}) "
              f"bbox_ratio={area_ratio:.2f}")

    n = len(per_image)
    if n == 0:
        print("ERROR: no images were evaluated.", file=sys.stderr)
        return 1

    report = {
        "what_this_measures": (
            "End-to-end accuracy of localisation + classification on "
            "photographs taken outside the PlantVillage distribution."
        ),
        "sample_size": {
            "images_evaluated": n,
            "images_listed_in_csv": len(rows),
            "warning": (
                f"n={n}. This is a smoke test, not a measurement. The "
                "confidence intervals below are wide enough to cover most "
                "outcomes you would care to distinguish. Do not quote these "
                "rates as if they were precise, and do not compare them to "
                "the PlantVillage numbers as if the difference were "
                "statistically established."
            ),
        },
        "config": {
            "checkpoint": resolution.path,
            "checkpoint_source": resolution.source,
            "confidence_threshold": args.confidence_threshold,
            "full_frame_ratio": args.full_frame_ratio,
        },
        "accuracy": rate_block(n_correct, n),
        "abstention": {
            **rate_block(n_below_threshold, n),
            "note": (
                "Fraction of photos where top-1 confidence fell below the "
                "threshold, so the app would show 'no confident "
                "classification' rather than a diagnosis. A high rate here on "
                "field photos is the pipeline behaving correctly, not failing."
            ),
        },
        "localisation": {
            **rate_block(n_full_frame, n),
            "note": (
                f"Fraction of photos where the bounding box covered >= "
                f"{args.full_frame_ratio:.0%} of the frame. The contour/HSV "
                "segmenter returns the whole frame when it cannot isolate "
                "anything, so a high rate here means stage 1 is effectively "
                "not running and the classifier is seeing an uncropped photo."
            ),
        },
        "confidence": {
            "correct_predictions": summarize(correct_conf),
            "incorrect_predictions": summarize(incorrect_conf),
            "comparison_note": (
                "Compare these against the same fields in the PlantVillage "
                "report. Confidence that stays high while accuracy drops is "
                "the dangerous pattern: it means the model is confidently "
                "wrong off-distribution and the confidence floor will not "
                "catch it."
            ),
        },
        "per_image": per_image,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    acc = report["accuracy"]
    lo, hi = acc["wilson_95_ci"]
    print("\n" + "=" * 62)
    print(f"  n = {n} photographs  (smoke test, not a measurement)")
    print(f"  accuracy            {acc['rate']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  below threshold     {n_below_threshold}/{n}")
    print(f"  full-frame bbox     {n_full_frame}/{n}  (localisation failures)")
    print("=" * 62)
    print(f"\nReport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
