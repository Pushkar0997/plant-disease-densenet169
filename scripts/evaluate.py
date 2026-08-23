"""
Evaluate a trained DenseNet-169 checkpoint against PlantVillage.

Run this on Colab, where the dataset and GPU are. It writes a JSON report you
can paste real numbers from into the README.

READ THIS BEFORE TRUSTING ANY NUMBER THIS SCRIPT PRINTS
-------------------------------------------------------

1. There is no sealed test set, and this script cannot create one.

   notebooks/plantvillage_densenet169_pipeline.ipynb (the notebook that
   actually trains) splits the data 80/20 into train and val only. The
   training loop saves a checkpoint every time val accuracy improves, and
   ReduceLROnPlateau also steers on val accuracy. So the val set drove
   checkpoint selection, and the train set was fitted directly. Every image
   in the dataset touched model selection through one path or the other.

   Carving a slice out of val does NOT produce a sealed partition — that
   slice still contributed to the val accuracy statistic that picked the
   checkpoint. This script therefore refuses to label anything "test". It
   reports on the reproduced val split and marks the result as
   selection-biased, which is honest, rather than inventing a clean number.

   To get a genuinely unbiased number you have to retrain with a three-way
   split. See --emit-split-plan and the README section "Getting an unbiased
   number".

2. The reported best val_acc=0.9686 is the maximum over 15 epochs on this
   same val split. Taking a max over 15 noisy estimates biases the number
   upward. The accuracy this script reports for the same split will usually
   be at or slightly below that figure, and is the better one to quote — but
   it is still not a held-out number, for the reason in (1).

3. The notebook's split is not exactly reproducible.

   Its build_train_val_split() creates one random.Random(42) and consumes it
   across classes in Path.iterdir() order, and shuffles each class's files in
   iterdir() order too. iterdir() returns arbitrary, filesystem-dependent
   order. So the exact train/val partition of the original run depends on
   directory ordering on that specific Colab runtime and cannot be recovered
   from the notebook alone.

   --split-order sorted (the default) gives a deterministic, reproducible
   split using the same seed, ratio and per-class logic. It will overlap
   heavily with the original split but is not guaranteed identical, so a
   handful of images that were in the original train set may land in val
   here. That leakage inflates accuracy slightly. The report records this.

   --split-order filesystem mimics the notebook exactly, including its
   nondeterminism. Use it only if you still have the original runtime.

4. Preprocessing mismatch between training and the app.

   The notebook validates with Resize((224,224)) — a straight squash, no
   crop. src/classifier.py infers with Resize(256) + CenterCrop(224), which
   rescales differently and cuts off the edges of the frame. Those are
   different input distributions, and the app's version is not the one the
   model was trained under.

   --preprocess notebook (default) matches training. --preprocess app matches
   what the Gradio app actually does at inference. Run both: the gap between
   them is the accuracy the deployed app is losing to this mismatch, and it
   is worth knowing before you publish a number measured one way and ship
   code that does it the other way.

Usage
-----
    python scripts/evaluate.py \
        --data-root /content/plantvillage_data/raw \
        --checkpoint /content/densenet169_plantvillage.pth \
        --class-mapping models/class_mapping.json \
        --output reports/plantvillage_eval.json
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifier import DiseaseClassifier  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Candidate confidence floors reported in the threshold sweep. Dense at the
# top end because a well-fit softmax classifier puts most correct predictions
# very close to 1.0, so the interesting trade-offs live above 0.9 — a sweep
# that stopped at 0.5 would miss the entire usable range.
SWEEP_THRESHOLDS = [
    0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
    0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999,
]


def find_class_root(root: Path) -> Path:
    """
    Locate the directory whose subfolders are the per-class image folders.

    Mirrors the notebook's find_class_root: Kaggle mirrors of PlantVillage
    nest the class folders at varying depths, so walk the tree and take the
    directory with the most image-containing subfolders.
    """
    candidates: List[Tuple[int, Path]] = []
    for dirpath, dirnames, _ in os.walk(root):
        if len(dirnames) < 2:
            continue
        sample_dir = Path(dirpath) / sorted(dirnames)[0]
        try:
            sample_files = os.listdir(sample_dir)[:20]
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        if any(Path(f).suffix.lower() in IMAGE_EXTENSIONS for f in sample_files):
            candidates.append((len(dirnames), Path(dirpath)))
    if not candidates:
        raise FileNotFoundError(f"No class-labelled image folders found under {root}")
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1]


def build_val_split(
    class_root: Path,
    val_split: float,
    seed: int,
    split_order: str,
) -> Tuple[List[Tuple[Path, str]], Dict[str, int]]:
    """
    Reproduce the notebook's per-class 80/20 split and return the val side.

    Faithfully mirrors the notebook's structure: a single random.Random(seed)
    shared across all classes, per-class shuffle, and split_idx computed as
    int(len(images) * (1 - val_split)) with the tail going to val.

    The only deliberate deviation is ordering. With split_order="sorted" both
    the class list and each class's file list are sorted before shuffling, so
    the result is deterministic across machines. With "filesystem" the raw
    iterdir() order is used, exactly as the notebook does — reproducing its
    behaviour including its nondeterminism. See the module docstring, note 3.
    """
    class_dirs = [d for d in class_root.iterdir() if d.is_dir()]
    if split_order == "sorted":
        class_dirs = sorted(class_dirs, key=lambda d: d.name)

    rng = random.Random(seed)
    val_items: List[Tuple[Path, str]] = []
    train_counts: Dict[str, int] = {}

    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if split_order == "sorted":
            images = sorted(images, key=lambda p: p.name)
        rng.shuffle(images)
        split_idx = int(len(images) * (1 - val_split))
        train_counts[class_dir.name] = split_idx
        for img in images[split_idx:]:
            val_items.append((img, class_dir.name))

    return val_items, train_counts


class EvalDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, str]], class_to_idx: Dict[str, int], transform):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, class_name = self.items[i]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.class_to_idx[class_name]


def build_transform(mode: str, image_size: int):
    """See module docstring note 4 for why these two differ and why it matters."""
    if mode == "notebook":
        resize = transforms.Resize((image_size, image_size))
        crop = []
    else:  # "app" — matches src/classifier.py
        resize = transforms.Resize(256)
        crop = [transforms.CenterCrop(image_size)]
    return transforms.Compose(
        [resize, *crop, transforms.ToTensor(),
         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )


def save_confusion_png(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.42)), dpi=140)
    # Row-normalized so large classes don't visually swamp small ones.
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "percentiles": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        # Percentiles matter more than mean/median for picking a floor: the
        # mean hides whether errors are spread out or bunched near the top.
        "percentiles": {
            f"p{q}": float(np.percentile(values, q))
            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
    }


def threshold_sweep(y_conf: np.ndarray, correct_mask: np.ndarray,
                    thresholds: List[float]) -> List[Dict[str, float]]:
    """
    For each candidate confidence floor, report what it actually buys.

    The confidence floor's whole job is to trade coverage for reliability:
    abstaining on some predictions so the ones still shown are more often
    right. Mean/median confidence alone can't tell you where to set it. This
    reports, at each threshold:

      errors_suppressed  - wrong predictions that fall below the floor, so
                           the user sees "not confident" instead of a wrong
                           diagnosis. This is the benefit.
      correct_lost       - correct predictions also pushed below the floor.
                           This is the cost.
      coverage           - fraction of images still given a diagnosis.
      selective_accuracy - accuracy among only those still diagnosed. This
                           is the number that matters to someone acting on
                           the app's output.

    Pick the floor where selective_accuracy is acceptable and coverage is
    still useful. Note that all of this is measured on PlantVillage, which
    is lab-condition data — a floor tuned here may behave very differently
    on field photographs. See scripts/evaluate_field.py.
    """
    n = int(y_conf.size)
    n_wrong = int((~correct_mask).sum())
    rows = []
    for t in thresholds:
        above = y_conf >= t
        n_above = int(above.sum())
        correct_above = int((above & correct_mask).sum())
        rows.append({
            "threshold": float(t),
            "diagnosed": n_above,
            "abstained": n - n_above,
            "coverage": (n_above / n) if n else None,
            "errors_suppressed": int(((~above) & (~correct_mask)).sum()),
            "errors_suppressed_pct_of_all_errors": (
                float(((~above) & (~correct_mask)).sum() / n_wrong) if n_wrong else None
            ),
            "correct_lost": int(((~above) & correct_mask).sum()),
            "selective_accuracy": (correct_above / n_above) if n_above else None,
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path, help="Root of the extracted PlantVillage data")
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to the trained .pth")
    p.add_argument("--class-mapping", type=Path, default=Path("models/class_mapping.json"))
    p.add_argument("--output", required=True, type=Path, help="Where to write the JSON report")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--val-split", type=float, default=0.2, help="Must match the training run (notebook uses 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Must match the training run (notebook uses 42)")
    p.add_argument("--split-order", choices=["sorted", "filesystem"], default="sorted")
    p.add_argument("--preprocess", choices=["notebook", "app"], default="notebook")
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}")

    # Load through DiseaseClassifier so this script exercises the same
    # checkpoint-loading and head-detection path the app uses. If that path
    # would silently fall back to an untrained head, we abort instead — a
    # metrics report generated from a random head is worse than no report.
    clf = DiseaseClassifier(
        weights_path=str(args.checkpoint),
        class_mapping_path=str(args.class_mapping),
        device=device,
    )
    if not clf.weights_loaded:
        print(f"\nERROR: checkpoint did not load: {clf.weights_load_error}", file=sys.stderr)
        print("Refusing to report metrics for an untrained model.", file=sys.stderr)
        return 1

    idx_to_class = clf.idx_to_class
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"[evaluate] {len(class_names)} classes from {args.class_mapping}")

    class_root = find_class_root(args.data_root)
    print(f"[evaluate] class root: {class_root}")

    on_disk = sorted(d.name for d in class_root.iterdir() if d.is_dir())
    missing = [c for c in class_names if c not in on_disk]
    extra = [c for c in on_disk if c not in class_to_idx]
    if missing or extra:
        # Mismatched folder names would silently mislabel everything.
        print("\nERROR: dataset folders do not match the class mapping.", file=sys.stderr)
        if missing:
            print(f"  In mapping but not on disk: {missing}", file=sys.stderr)
        if extra:
            print(f"  On disk but not in mapping: {extra}", file=sys.stderr)
        print("\nThis usually means the checkpoint was trained on a different "
              "PlantVillage mirror than the one downloaded here. Note that the "
              "'emmarex/plantdisease' Kaggle mirror used by the training "
              "notebook has 15 classes, not the full 38.", file=sys.stderr)
        return 1

    val_items, train_counts = build_val_split(class_root, args.val_split, args.seed, args.split_order)
    print(f"[evaluate] val images: {len(val_items)} (train side: {sum(train_counts.values())})")

    transform = build_transform(args.preprocess, args.image_size)
    loader = DataLoader(
        EvalDataset(val_items, class_to_idx, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = clf.model.to(device).eval()
    all_true, all_pred, all_conf = [], [], []

    with torch.no_grad():
        for bi, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            probs = F.softmax(model(images).float(), dim=1)
            conf, pred = probs.max(dim=1)
            all_true.append(labels.numpy())
            all_pred.append(pred.cpu().numpy())
            all_conf.append(conf.cpu().numpy())
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(loader)}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_conf = np.concatenate(all_conf)

    labels_range = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)

    correct_mask = y_true == y_pred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".confusion.png")
    save_confusion_png(cm, class_names, png_path)

    # Dump per-prediction rows. The summary JSON only holds aggregates, so
    # without this any future re-analysis (a different threshold sweep, a
    # calibration curve, per-class error inspection) would mean re-running
    # the whole evaluation — which on Colab means re-downloading the dataset.
    # This file makes the run reusable.
    predictions_path = args.output.with_suffix(".predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_class", "predicted_class", "correct", "confidence"])
        for i in range(len(y_true)):
            writer.writerow([
                i,
                class_names[y_true[i]],
                class_names[y_pred[i]],
                int(correct_mask[i]),
                f"{y_conf[i]:.6f}",
            ])
    print(f"[evaluate] per-prediction CSV: {predictions_path}")

    n_wrong = int((~correct_mask).sum())

    report = {
        "what_this_measures": (
            "Accuracy on the reproduced validation split of the training run. "
            "This is NOT a held-out test number — the val split drove "
            "checkpoint selection. See 'caveats'."
        ),
        "config": {
            "checkpoint": str(args.checkpoint),
            "class_mapping": str(args.class_mapping),
            "data_root": str(args.data_root),
            "class_root": str(class_root),
            "num_classes": len(class_names),
            "val_split": args.val_split,
            "seed": args.seed,
            "split_order": args.split_order,
            "preprocess": args.preprocess,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "device": device,
        },
        "dataset": {
            "val_images": int(len(val_items)),
            "train_images_excluded": int(sum(train_counts.values())),
        },
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        },
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels_range
        },
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
            "png": str(png_path),
        },
        "confidence": {
            "correct_predictions": summarize(y_conf[correct_mask]),
            "incorrect_predictions": summarize(y_conf[~correct_mask]),
            "note": (
                "If these two distributions overlap heavily, no single "
                "confidence threshold can separate right from wrong answers, "
                "and the app's confidence floor is doing less than it appears."
            ),
        },
        "threshold_sweep": {
            "note": (
                "What each candidate confidence floor actually buys. Use this "
                "to choose DEFAULT_CONFIDENCE_FLOOR in app.py instead of "
                "guessing. Measured on PlantVillage (lab-condition data) — a "
                "floor tuned here may behave differently on field photos."
            ),
            "rows": threshold_sweep(y_conf, correct_mask, SWEEP_THRESHOLDS),
        },
        "caveats": [
            "NOT A HELD-OUT TEST SET. The notebook splits train/val only, and "
            "saved a checkpoint on every val-accuracy improvement, so val drove "
            "model selection. No partition of this dataset is sealed.",
            "The reported best val_acc=0.9686 is a maximum over 15 epochs on "
            "this same split and is optimistically biased. The accuracy above "
            "is a single-checkpoint estimate on the same data — better, but "
            "still not held out.",
            f"split_order={args.split_order}. The notebook's split depends on "
            "Path.iterdir() ordering and cannot be exactly reproduced. Some "
            "images that were in the original train set may appear here, which "
            "inflates accuracy by an unknown but probably small amount.",
            f"preprocess={args.preprocess}. The notebook trains/validates with "
            "Resize((224,224)); src/classifier.py infers with Resize(256)+"
            "CenterCrop(224). Run both modes — the difference is what the "
            "deployed app loses to that mismatch.",
            "PlantVillage is lab-condition imagery (single leaf, uniform "
            "background, controlled light). This number does not predict field "
            "performance. Use scripts/evaluate_field.py for that.",
        ],
        "to_get_an_unbiased_number": (
            "Retrain with a three-way split: carve a test partition first, "
            "never load it during training, select the checkpoint on val, then "
            "evaluate once on test. Nothing computed from the current run can "
            "substitute for this."
        ),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 62)
    print(f"  accuracy     {report['overall']['accuracy']:.4f}")
    print(f"  macro F1     {report['overall']['macro_f1']:.4f}")
    print(f"  weighted F1  {report['overall']['weighted_f1']:.4f}")
    print("=" * 62)
    print("  This is a validation-split number, not a test number.")
    print("  It is selection-biased. See 'caveats' in the JSON.")
    print("=" * 62)
    print("\n  Confidence floor trade-off (PlantVillage):")
    print(f"    {'floor':>7}  {'coverage':>9}  {'sel.acc':>8}  {'errors hidden':>14}  {'correct lost':>12}")
    for row in report["threshold_sweep"]["rows"]:
        if row["threshold"] < 0.30:
            continue
        sel = f"{row['selective_accuracy']:.4f}" if row["selective_accuracy"] is not None else "n/a"
        print(f"    {row['threshold']:>7.3f}  {row['coverage']:>8.1%}  {sel:>8}  "
              f"{row['errors_suppressed']:>6}/{n_wrong:<7}  {row['correct_lost']:>12}")
    print("\n  Pick the floor where selective accuracy is acceptable and")
    print("  coverage is still useful. Set it as DEFAULT_CONFIDENCE_FLOOR in app.py.")

    print(f"\nReport: {args.output}\nConfusion matrix: {png_path}\nPredictions CSV: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
