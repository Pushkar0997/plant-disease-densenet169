# plant-disease-densenet169

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A two-stage leaf disease classifier: OpenCV contour/HSV segmentation localises a
leaf in the frame, then a fine-tuned DenseNet-169 classifies the cropped region.
A Gradio app wraps both stages.

This is a learning/research project. It is not validated for agricultural
decision-making, and the "Known limitations" section below is the most important
part of this file.

## Measured Evaluation Metrics (PlantVillage Dataset)

The metrics below were measured on the validation partition (4,134 images across 15 classes) using [`scripts/evaluate.py`](scripts/evaluate.py) on the fine-tuned DenseNet-169 checkpoint. Full evaluation outputs and per-class statistics are recorded in [`reports/plantvillage_eval.json`](reports/plantvillage_eval.json).

| Metric | Value | Produced by / Source |
| :--- | :--- | :--- |
| **Validation accuracy (single checkpoint)** | **97.80%** (4,043 / 4,134) | `reports/plantvillage_eval.json` |
| **Macro F1** | **0.9765** (97.65%) | `reports/plantvillage_eval.json` |
| **Weighted F1** | **0.9780** (97.80%) | `reports/plantvillage_eval.json` |
| **Mean top-1 confidence (correct predictions)** | **98.06%** (median: 99.98%) | `reports/plantvillage_eval.json` |
| **Mean top-1 confidence (incorrect predictions)** | **69.57%** (median: 68.20%) | `reports/plantvillage_eval.json` |
| **Field-photo accuracy** | Not measured | `scripts/evaluate_field.py` |
| **Localisation failure rate on field photos** | Not measured | `scripts/evaluate_field.py` |

### Per-Class Performance Breakdown (15 Classes)

| Class Name | Precision | Recall | F1-Score | Support (Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Pepper (bell): Bacterial spot** | 0.9950 | 0.9900 | **0.9925** | 200 |
| **Pepper (bell): Healthy** | 0.9966 | 0.9966 | **0.9966** | 296 |
| **Potato: Early blight** | 1.0000 | 0.9900 | **0.9950** | 200 |
| **Potato: Late blight** | 0.9754 | 0.9900 | **0.9826** | 200 |
| **Potato: Healthy** | 1.0000 | 0.9677 | **0.9836** | 31 |
| **Tomato: Bacterial spot** | 0.9976 | 0.9883 | **0.9929** | 426 |
| **Tomato: Early blight** | 0.9048 | 0.9500 | **0.9268** | 200 |
| **Tomato: Late blight** | 0.9917 | 0.9398 | **0.9651** | 382 |
| **Tomato: Leaf Mold** | 0.9641 | 0.9843 | **0.9741** | 191 |
| **Tomato: Septoria leaf spot** | 0.9488 | 0.9915 | **0.9697** | 355 |
| **Tomato: Spider mites (Two-spotted)** | 0.9354 | 0.9911 | **0.9624** | 336 |
| **Tomato: Target Spot** | 0.9807 | 0.9039 | **0.9407** | 281 |
| **Tomato: Yellow Leaf Curl Virus** | 0.9969 | 0.9953 | **0.9961** | 642 |
| **Tomato: Mosaic virus** | 0.9615 | 1.0000 | **0.9804** | 75 |
| **Tomato: Healthy** | 0.9968 | 0.9812 | **0.9889** | 319 |
| **Overall Dataset Total** | — | — | **0.9780** | **4,134** |

> **Confusion Matrix**: Visualized matrix plot is saved at [`reports/plantvillage_eval.confusion.png`](reports/plantvillage_eval.confusion.png).

The training notebook printed `Best val_acc=0.9686`. That number is **not**
reported above as a result, for two reasons:

1. It is the maximum over 15 epochs on the split that was used to choose which
   checkpoint to keep. Taking a maximum over 15 noisy estimates biases the value
   upward; it is a selection statistic, not a performance estimate.
2. The same split drove `ReduceLROnPlateau` and the save-on-improvement logic,
   so it influenced training as well as selection.

`scripts/evaluate.py` reports a single-checkpoint number on the same split,
which removes the max-over-epochs bias but not the selection bias. There is no
sealed partition of this dataset. See "Getting an unbiased number".

## Class coverage

The class list is read at runtime from `models/class_mapping.json`; the app and
both evaluation scripts derive their class count from that file rather than
assuming a number.

**The checked-in mapping has 15 classes, not 38.** It covers Pepper (bell),
Potato and Tomato only. The training notebook downloads the Kaggle mirror
`emmarex/plantdisease`, which is the 15-class subset of PlantVillage, not the
full 38-class release. Earlier versions of this README and the app claimed 38
classes; that claim was wrong. If you retrain on the full 38-class dataset,
re-export the mapping and the app will pick up the change automatically.

## Model weights

The checkpoint is roughly 50 MB and is not in this repository — `.gitignore`
excludes `*.pth`. It is resolved at startup by `src/weights.py` in this order:

1. `models/densenet169_plant_disease.pth`, if present. A local file always wins,
   so a development loop never needs the network.
2. Hugging Face Hub, via `huggingface_hub.hf_hub_download`, cached locally so it
   downloads once.

Configure the Hub source with environment variables:

```bash
export PLANT_DISEASE_HF_REPO_ID="<your-username>/plant-disease-densenet169"
export PLANT_DISEASE_HF_FILENAME="densenet169_plant_disease.pth"   # optional
```

The default repo ID is the literal placeholder `CHANGE_ME/plant-disease-densenet169`.
It is a placeholder on purpose: a plausible-looking default would fail with a 404
that is hard to distinguish from "not configured yet".

**If no checkpoint resolves, the app does not fall back to something usable.**
The DenseNet-169 backbone is ImageNet-pretrained but the classification head is
randomly initialised until a checkpoint loads. Without one the model emits noise.
The app starts, shows a persistent warning banner, and replaces every result with
an explicit "untrained model" notice instead of a diagnosis. See "The bug this
audit fixed".

## Setup

```bash
git clone https://github.com/Pushkar0997/plant-disease-densenet169.git
cd plant-disease-densenet169
python -m venv venv && source venv/bin/activate    # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The app serves at `http://127.0.0.1:7860`.

## Evaluation scripts

Both are meant to run on Colab, where the dataset and GPU are. Both refuse to
emit metrics if the checkpoint fails to load, rather than reporting numbers for
an untrained model.

### `scripts/evaluate.py` — PlantVillage

```bash
python scripts/evaluate.py \
    --data-root /content/plantvillage_data/raw \
    --checkpoint /content/densenet169_plantvillage.pth \
    --class-mapping models/class_mapping.json \
    --output reports/plantvillage_eval.json
```

Reports overall accuracy, macro and weighted F1, per-class precision/recall/F1,
a confusion matrix as JSON and PNG, and mean/median top-1 confidence split by
whether the prediction was correct. Everything lands in the JSON.

Two flags change what is being measured and are worth running both ways:

- `--split-order sorted|filesystem`. The notebook's split is not exactly
  reproducible (see "Known limitations"). `sorted` is deterministic; `filesystem`
  mimics the notebook including its nondeterminism.
- `--preprocess notebook|app`. The notebook trains and validates with
  `Resize((224,224))`; `src/classifier.py` infers with `Resize(256) +
  CenterCrop(224)`. These are different input distributions. The gap between the
  two modes is what the deployed app loses to that mismatch.

### `scripts/evaluate_field.py` — field photographs

```bash
python scripts/evaluate_field.py \
    --images-dir data/field \
    --labels-csv data/field/labels.csv \
    --checkpoint /content/densenet169_plantvillage.pth \
    --output reports/field_eval.json
```

Runs the **full** pipeline, localisation included, because stage 1 is the part
most likely to fail on real photos. The labels CSV is `filename,label` with a
header, where `label` exactly matches a class name in the mapping.

Reports accuracy, the confidence distribution for comparison against the
PlantVillage numbers, and how often localisation returned a box covering most of
the frame — the contour segmenter's failure signature, meaning stage 1 is
effectively not running.

Every rate carries a Wilson 95% confidence interval. At 20-30 photos those
intervals are wide: at n=25, an observed 68% spans roughly 48-83%. The script
prints a sample-size warning in its output and its JSON. This is a smoke test,
not a measurement.

## Getting an unbiased number

Nothing computed from the current training run is a held-out estimate. The
notebook splits train/val only; the training loop saved a checkpoint on every
val-accuracy improvement and stepped the LR scheduler on the same signal. Every
image influenced the result through training or through selection.

Slicing a piece out of val does not fix this — that slice still contributed to
the accuracy statistic that picked the checkpoint. `scripts/evaluate.py`
therefore never labels anything "test".

The only fix is to retrain: carve a test partition first, never load it during
training, select on val, evaluate once on test. Until that happens, describe the
number as validation accuracy and say it is selection-biased.

## The bug this audit fixed

`app.py` pointed at `models/densenet169_plant_disease.pth`, which `.gitignore`
excludes, so a fresh clone had no checkpoint. `src/classifier.py` handled the
missing file by printing "Running with pre-configured weights" and continuing —
but nothing was pre-configured. `_build_model` attaches a randomly initialised
`Dropout + Linear` head to the ImageNet backbone.

A fresh clone therefore ran a random 38-way head over ImageNet features and
presented the output as a diagnosis with a red "PATHOLOGY DETECTED" badge and
treatment advice. An observed example: "Pepper Bell: Bacterial Spot" at 10.0%
confidence, against a 1/15 ≈ 6.7% chance baseline for the checked-in mapping.

The misleading badge was not only in the report text. `Visualizer.draw_overlay`
rendered the red status directly into the annotated image pixels, so fixing the
markdown alone would have left a confidently-coloured image.

What changed:

- `DiseaseClassifier.weights_loaded` is authoritative and is returned from
  `predict()`. The misleading log message is replaced with an explicit warning.
- `PlantDiagnosticPipeline.run` computes a `status` of `diagnosed`,
  `low_confidence` or `untrained`. `is_healthy` is `None` and agronomic advice is
  replaced with a safety explanation for anything that is not `diagnosed`.
- `Visualizer.draw_overlay` takes that status instead of an `is_healthy` boolean,
  so an untrained or low-confidence result is structurally incapable of getting a
  red/green health badge.
- The app shows a persistent startup banner when no checkpoint is loaded.
- `dry_run.py` had the same bug and got the same fix.
- Checkpoint loading now auto-detects the head layout and refuses to load a
  checkpoint whose class count disagrees with the mapping.

## Known limitations

**PlantVillage is lab-condition imagery.** Single leaves, uniform backgrounds,
controlled lighting. Accuracy on it does not transfer to field photographs with
soil, sky, overlapping leaves and hard shadows. The two-stage architecture exists
because field photos are messy, which means the PlantVillage number does not
measure the thing the pipeline was built for. This is what
`scripts/evaluate_field.py` is for.

**Localisation defaults to contour/HSV segmentation.** `LeafDetector` uses
OpenCV HSV masking plus contour analysis unless a `yolo_model_path` is passed
explicitly. Nothing in `app.py` passes one, so the default path is always
contour segmentation. YOLOv8 is optional and off by default. Even when enabled it
would be COCO-pretrained, and COCO has no leaf or plant class — the training
notebook says as much in its own notes. Descriptions of this project as
"YOLOv8-based" would be inaccurate.

**The segmenter's confidence score is not a confidence.**
`_contour_segmentation_detection` computes it as
`clip(0.65 + area_ratio * 0.3, 0.50, 0.95)` — a function of bounding box area,
not of any evidence that a leaf was found. A larger box scores higher, so the
score is highest exactly when segmentation has failed and returned most of the
frame. The "Localization Confidence Threshold" slider does not filter on it.

**The confidence floor is calibrated on lab data only.** The default floor
(`DEFAULT_CONFIDENCE_FLOOR = 0.80` in `app.py`) was selected from the threshold
sweep in `reports/plantvillage_eval.json` rather than guessed. The previous 0.40
default suppressed zero of the 91 validation errors while costing 2 correct
predictions — it was doing nothing. At 0.80, coverage is 95.3% (3,938 of 4,134
images diagnosed), selective accuracy is 99.31% (up from 97.80%), and 64 of 91
validation errors are suppressed at a cost of 132 correct predictions
reclassified as unconfident. However, the sweep was measured on PlantVillage,
which is lab-condition data where correct predictions cluster near 1.0, so the
floor may abstain considerably more often on field photographs — and this is
untested because `scripts/evaluate_field.py` has not been run. The floor remains
adjustable in the Gradio UI and via script flags.

**Training and inference preprocess differently.** Training uses
`Resize((224,224))`; inference uses `Resize(256) + CenterCrop(224)`, which
rescales differently and crops the frame edges. Unquantified until
`scripts/evaluate.py --preprocess` is run both ways.

**Two notebooks, only one of which trains.**
`notebooks/plantvillage_densenet169_pipeline.ipynb` is the working one: it
downloads the data, builds the split, and runs the training loop. It freezes the
backbone and trains only the head, with a `Linear(1664,512) → ReLU → Dropout →
Linear(512,N)` classifier.

`notebooks/01_densenet169_training_pipeline.ipynb` is a skeleton. It defines
transforms, a model, an optimizer and train/validate functions, but contains no
dataset loading, no dataloaders and no epoch loop. Its final cell saves
`model.state_dict()` — so running it top to bottom writes an **untrained** head
to `models/densenet169_plant_disease.pth`, which is precisely the failure this
audit is about. It also builds a different head (`Dropout → Linear`) and declares
`num_classes: 38`, matching neither the other notebook nor the checked-in
mapping. Treat it as unused, or delete it.

Checkpoint loading now detects which of the two head layouts a file contains and
adapts, so either notebook's output loads. A class-count disagreement between
checkpoint and mapping is refused outright, since it would silently attach wrong
disease names to predictions.

**Observed failure modes.** Running the app without a checkpoint produced a
confident-looking "Pepper Bell: Bacterial Spot — PATHOLOGY DETECTED" at 10.0%
confidence. This is fixed, but it is the concrete illustration of why the
untrained state now fails loudly.

## Repository layout

```
app.py                      Gradio interface
dry_run.py                  CLI smoke test on data/samples
src/
  detector.py               Stage 1: contour/HSV localisation, optional YOLOv8
  classifier.py             Stage 2: DenseNet-169, checkpoint loading, head detection
  pipeline.py               Coordinates both stages, computes diagnosis status
  visualizer.py             Bounding box overlay and top-k chart
  weights.py                Local-then-Hub checkpoint resolution
scripts/
  evaluate.py               PlantVillage metrics report
  evaluate_field.py         Field-photo metrics report
models/class_mapping.json   Index to class name (15 classes)
notebooks/                  See "Two notebooks" above
```

## License

MIT. See [LICENSE](LICENSE).
