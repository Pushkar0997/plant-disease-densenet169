---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Plant Disease Detection (DenseNet-169)

Two-stage leaf disease classifier: OpenCV contour/HSV segmentation localises a
leaf, then a fine-tuned DenseNet-169 classifies the cropped region across 15
PlantVillage classes (Pepper bell, Potato, Tomato).

This is a learning/research project. It is **not validated for agricultural
decision-making.** Read the limitations below before acting on any output.

## Measured performance

| Metric | Value |
| :--- | :--- |
| Validation accuracy | 97.80% (4,043 / 4,134) |
| Macro F1 | 0.9765 |
| Weighted F1 | 0.9780 |
| Field-photo accuracy | Not measured |

**These numbers are validation-split accuracy, not held-out test accuracy.**
The training run split train/val only and saved a checkpoint on every val
improvement, so the val set drove model selection. No partition of the data is
sealed. The number is real but optimistically biased.

## Limitations

**PlantVillage is lab-condition imagery.** Single leaves, uniform backgrounds,
controlled lighting. The 97.8% figure does not transfer to field photographs
with soil, sky, overlapping leaves and hard shadows. Field performance has not
been measured.

**Confidence floor set to 0.80.** Below that, the app reports "no confident
classification" rather than a diagnosis. Chosen from a threshold sweep on
validation data: at 0.80, coverage is 95.3% and accuracy among diagnosed images
rises to 99.31%. Tuned on lab-condition data, so it may abstain more often on
real photographs.

**Localisation uses contour/HSV segmentation**, not YOLOv8. The segmenter
returns the whole frame when it cannot isolate a leaf, and its reported
"confidence" is a function of bounding-box area rather than evidence a leaf was
found.

**15 classes, not 38.** The model was trained on the `emmarex/plantdisease`
Kaggle mirror, which covers Pepper (bell), Potato and Tomato only.

## Source

Full code, evaluation scripts and detailed limitations:
https://github.com/Pushkar0997/plant-disease-densenet169
