---
license: mit
tags:
  - image-classification
  - plant-disease
  - densenet
  - pytorch
library_name: pytorch
pipeline_tag: image-classification
---

# DenseNet-169 Plant Disease Classifier (PlantVillage, 15 classes)

Fine-tuned DenseNet-169 checkpoint for leaf disease classification across 15
PlantVillage classes covering Pepper (bell), Potato and Tomato.

Used by: https://github.com/Pushkar0997/plant-disease-densenet169

## Files

| File | Description |
| :--- | :--- |
| `densenet169_plant_disease.pth` | Model state_dict (~54 MB) |

## Architecture

ImageNet-pretrained DenseNet-169 backbone with the classifier head replaced.
The backbone was frozen; only the head was trained. Input is 224x224 RGB,
normalised with ImageNet mean/std.

Note that the training notebook and the inference code originally defined
different head shapes. The loader in `src/classifier.py` auto-detects the
head layout from the checkpoint's `state_dict` keys, so either variant loads.

## Measured performance

| Metric | Value |
| :--- | :--- |
| Validation accuracy | 97.80% (4,043 / 4,134) |
| Macro F1 | 0.9765 |
| Weighted F1 | 0.9780 |
| Mean top-1 confidence, correct | 98.06% |
| Mean top-1 confidence, incorrect | 69.57% |

**This is validation-split accuracy, not held-out test accuracy.** The
training run split train/val only and saved a checkpoint on every validation
improvement, so the validation set drove model selection. No partition of the
dataset is sealed. The figure is real but optimistically biased.

Full evaluation output, including per-class metrics and a confusion matrix:
`reports/plantvillage_eval.json` in the GitHub repository.

## Limitations

**Trained on lab-condition imagery.** PlantVillage images are single leaves on
uniform backgrounds under controlled lighting. Performance on field
photographs — soil, sky, overlapping leaves, hard shadows — has not been
measured and should not be assumed to match.

**15 classes, not 38.** Trained on the `emmarex/plantdisease` Kaggle mirror,
which covers Pepper (bell), Potato and Tomato only. Passing an image of any
other crop will still produce a confident-looking prediction from those 15
options.

**Poorly calibrated.** Correct predictions cluster near 1.0 confidence, and
incorrect ones average around 0.70. The consuming application applies a 0.80
confidence floor, chosen from a threshold sweep, below which it reports "no
confident classification" instead of a diagnosis.

**Not validated for agricultural decision-making.** This is a learning
project. Do not treat its output as a basis for treating a real crop.

## Usage

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "PushkarKumar/plant-disease-densenet169",
    "densenet169_plant_disease.pth",
)
```

Class index to label mapping is in `models/class_mapping.json` in the GitHub
repository. The checkpoint's output ordering matches that file.
