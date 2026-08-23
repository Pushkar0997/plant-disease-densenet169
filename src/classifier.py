"""
Plant Disease Classifier using DenseNet169 Transfer Learning.
Supports PyTorch model creation, weights loading, preprocessing, and top-k inference.
"""

import json
import os
import re
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class DiseaseClassifier:
    """
    Fine-tuned DenseNet169 classifier for multi-class plant leaf disease diagnostics.
    """

    DEFAULT_CLASS_NAMES = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

    DISEASE_DESCRIPTIONS = {
        "Apple_scab": "Fungal infection causing olive-green to dark brown spots on leaves and fruit. Use preventative fungicides like Captan or copper-based sprays.",
        "Black_rot": "Fungal disease causing circular brown lesions with concentric rings. Prune infected stems and ensure good air circulation.",
        "Cedar_apple_rust": "Fungal pathogen that alternates between junipers and apple trees. Look for bright orange-yellow spots on upper leaf surfaces.",
        "Powdery_mildew": "White talcum-powder-like fungal coating on foliage. Apply sulfur sprays, neem oil, or bio-fungicides.",
        "Cercospora_leaf_spot Gray_leaf_spot": "Rectangular tan-to-gray lesions bounded by leaf veins. Rotate crops and apply strobilurin or triazole fungicides.",
        "Common_rust_": "Small cinnamon-brown pustules on upper and lower leaf surfaces. Utilize resistant hybrids and timely fungicide application.",
        "Northern_Leaf_Blight": "Long, elliptical grayish-green cigar-shaped lesions. Plant resistant cultivars and manage residue.",
        "Esca_(Black_Measles)": "Complex grapevine trunk disease showing tiger-stripe chlorosis on foliage. Practice clean pruning and avoid large wounds.",
        "Leaf_blight_(Isariopsis_Leaf_Spot)": "Angular red-brown spots with dark borders. Apply protective copper fungicides before bloom.",
        "Haunglongbing_(Citrus_greening)": "Bacterial disease spread by psyllid vectors causing asymmetric blotchy mottle. Manage psyllid populations and remove infected trees.",
        "Bacterial_spot": "Small water-soaked circular lesions turning brown with yellow halos. Use copper-mancozeb sprays and avoid overhead watering.",
        "Early_blight": "Brown spots with concentric 'target-board' rings on older leaves. Mulch soil, prune lower leaves, and apply copper fungicides.",
        "Late_blight": "Destructive water-soaked dark lesions with white mold on leaf undersides in humid conditions. Apply protective fungicides immediately.",
        "Leaf_Mold": "Pale yellow patches on upper surface with olive-velvety fungal growth underneath. Improve greenhouse ventilation and reduce humidity.",
        "Septoria_leaf_spot": "Abundant small circular spots with gray centers and dark borders on lower leaves. Avoid splashing water and rotate crops.",
        "Spider_mites Two-spotted_spider_mite": "Fine yellow stippling on leaves with fine webbing underneath. Treat with insecticidal soaps or predatory mites.",
        "Target_Spot": "Brown lesions with noticeable concentric rings and distinct halos. Apply chlorothalonil or azoxystrobin fungicides.",
        "Tomato_Yellow_Leaf_Curl_Virus": "Viral disease transmitted by whiteflies causing upward leaf cupping, stunted growth, and chlorosis. Use insect netting and resistant cultivars.",
        "Tomato_mosaic_virus": "Mottled light/dark green mosaic patterns on leaves with distortion. Disinfect tools and avoid tobacco handling around crops.",
        "Leaf_scorch": "Irregular purple or brown blotches on foliage that coalesce. Remove diseased leaves and optimize drip irrigation.",
        "healthy": "Foliage is vigorous, vibrant, and shows no pathological symptoms. Continue routine maintenance, balanced nutrition, and monitoring."
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the DenseNet169 disease classifier.

        Args:
            weights_path: Optional path to saved PyTorch weights (.pth / .pt).
            class_mapping_path: Optional path to JSON file containing class index mapping.
            num_classes: Number of output classes (defaults to 38 if not specified).
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load class mappings
        self.idx_to_class = self._load_class_mapping(class_mapping_path)
        self.num_classes = num_classes or len(self.idx_to_class)

        # Build DenseNet169 model architecture. NOTE: the DenseNet backbone is
        # ImageNet-pretrained, but the classification head below is always
        # randomly initialized until a checkpoint is successfully loaded via
        # load_weights(). Predictions from this head are meaningless noise
        # over `num_classes` categories unless self.weights_loaded is True.
        self.model = self._build_model(self.num_classes)

        # `weights_loaded` is the single source of truth for whether this
        # classifier is producing real predictions or random-head noise.
        # Every caller (app.py, dry_run.py, scripts/evaluate*.py) MUST check
        # this before treating a prediction as a diagnosis.
        self.weights_loaded = False
        self.weights_load_error: Optional[str] = None

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        elif weights_path:
            self.weights_load_error = f"Weights file not found at {weights_path}."
            print(
                f"[DiseaseClassifier] WARNING: {self.weights_load_error} "
                "No checkpoint was loaded. This classifier is running with an "
                "UNTRAINED, randomly initialized classification head. Its "
                "predictions are noise, not diagnoses. weights_loaded=False."
            )
        else:
            self.weights_load_error = "No weights_path was provided."
            print(
                "[DiseaseClassifier] WARNING: No weights_path was provided. "
                "This classifier is running with an UNTRAINED, randomly "
                "initialized classification head. Its predictions are noise, "
                "not diagnoses. weights_loaded=False."
            )

        self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_class_mapping(self, mapping_path: Optional[str]) -> Dict[int, str]:
        """Loads index-to-class dictionary."""
        if mapping_path and os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    raw_mapping = json.load(f)
                    return {int(k): str(v) for k, v in raw_mapping.items()}
            except Exception as e:
                print(f"[DiseaseClassifier] Could not parse mapping from {mapping_path}: {e}")

        return {i: name for i, name in enumerate(self.DEFAULT_CLASS_NAMES)}

    def _build_model(self, num_classes: int) -> nn.Module:
        """Instantiates DenseNet169 with custom classification head."""
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        num_features = model.classifier.in_features

        # Replace classification layer with customized Dropout + Linear head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
        return model

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Optional[Dict[str, Any]]:
        """Pulls the raw tensor state_dict out of the various checkpoint layouts."""
        if not isinstance(checkpoint, dict):
            return None
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint

    @staticmethod
    def _build_head_from_state_dict(state_dict: Dict[str, Any], num_features: int) -> nn.Module:
        """
        Rebuilds the classifier head to match the shapes actually present in a
        checkpoint, instead of assuming the Dropout+Linear head that
        _build_model() creates.

        This exists because the repo contains two different head definitions:

          * src/classifier.py / notebooks/01_...ipynb:
                Sequential(Dropout(0.3), Linear(1664, N))
                -> keys classifier.1.weight / classifier.1.bias
          * notebooks/plantvillage_densenet169_pipeline.ipynb (the one that
            actually trains):
                Sequential(Linear(1664, 512), ReLU(), Dropout(0.3), Linear(512, N))
                -> keys classifier.0.* and classifier.3.*

        Loading a checkpoint of the second kind into a head of the first kind
        raises a shape/key error, which (before this change) got swallowed and
        left an untrained random head in place. Detecting the layout from the
        checkpoint itself means either notebook's output loads correctly.

        Raises ValueError on any head layout that isn't one of the two above,
        rather than guessing — a wrong guess here reintroduces exactly the
        silent-garbage failure mode this audit is trying to remove.
        """
        linear_layers = sorted(
            (int(k.split(".")[1]), tuple(v.shape))
            for k, v in state_dict.items()
            if k.startswith("classifier.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 2
        )

        if not linear_layers:
            raise ValueError(
                "Checkpoint contains no 'classifier.*.weight' 2-D tensors; its "
                "classification head could not be identified."
            )

        if len(linear_layers) == 1:
            idx, (out_f, in_f) = linear_layers[0]
            if in_f != num_features:
                raise ValueError(
                    f"Checkpoint head expects {in_f} input features but the "
                    f"DenseNet-169 backbone produces {num_features}."
                )
            if idx == 1:
                return nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, out_f))
            if idx == 0:
                return nn.Sequential(nn.Linear(in_f, out_f))
            raise ValueError(f"Unrecognized single-Linear head at classifier.{idx}.")

        if len(linear_layers) == 2:
            (idx_a, (hidden, in_f)), (idx_b, (out_f, hidden_b)) = linear_layers
            if (idx_a, idx_b) != (0, 3) or hidden != hidden_b:
                raise ValueError(
                    f"Unrecognized two-Linear head layout: indices {idx_a},{idx_b} "
                    f"with shapes {hidden}x{in_f} and {out_f}x{hidden_b}."
                )
            if in_f != num_features:
                raise ValueError(
                    f"Checkpoint head expects {in_f} input features but the "
                    f"DenseNet-169 backbone produces {num_features}."
                )
            return nn.Sequential(
                nn.Linear(in_f, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, out_f),
            )

        raise ValueError(
            f"Checkpoint head has {len(linear_layers)} Linear layers; only the "
            "1- and 2-Linear layouts used in this repo are supported."
        )

    def load_weights(self, weights_path: str):
        """
        Loads state dict into model.

        On failure this does NOT raise: it records the reason in
        `self.weights_load_error` and leaves `self.weights_loaded` False, so
        the model keeps whatever head it had before (random, if this was
        called from __init__). Callers that need a hard failure instead of a
        silent untrained fallback (e.g. scripts/evaluate.py, which must never
        report metrics for a model that didn't actually load) should check
        `weights_loaded` / `weights_load_error` after calling this and abort
        themselves — see scripts/evaluate.py for that pattern.
        """
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = self._extract_state_dict(checkpoint)

            if state_dict is None:
                # A pickled whole-model object rather than a state_dict.
                self.model = checkpoint
            else:
                num_features = 1664  # DenseNet-169 final feature width
                head = self._build_head_from_state_dict(state_dict, num_features)

                # The checkpoint decides the class count, not the mapping file.
                # If they disagree, the labels shown to the user would be
                # silently wrong (index N meaning a different disease than the
                # mapping claims), so refuse rather than mislabel.
                checkpoint_classes = head[-1].out_features
                if checkpoint_classes != len(self.idx_to_class):
                    raise ValueError(
                        f"Checkpoint has {checkpoint_classes} output classes but "
                        f"the loaded class mapping has {len(self.idx_to_class)} "
                        "entries. Refusing to load: predictions would be labelled "
                        "with the wrong disease names. Point class_mapping_path at "
                        "the mapping exported by the same training run as this "
                        "checkpoint."
                    )

                self.model.classifier = head
                self.model.load_state_dict(state_dict, strict=True)
                self.num_classes = checkpoint_classes

            self.weights_loaded = True
            self.weights_load_error = None
            print(f"[DiseaseClassifier] Successfully loaded weights from {weights_path}")
        except Exception as e:
            self.weights_loaded = False
            self.weights_load_error = f"Failed to load checkpoint {weights_path}: {e}"
            print(
                f"[DiseaseClassifier] ERROR: {self.weights_load_error}\n"
                "[DiseaseClassifier] The classification head remains UNTRAINED. "
                "weights_loaded=False. This is commonly a shape/key mismatch "
                "between the saved state_dict and _build_model()'s head — "
                "check that the checkpoint's classifier architecture matches "
                "the Dropout+Linear head defined here."
            )

    KNOWN_CROPS = [
        "Pepper__bell", "Pepper bell", "Pepper,_bell", "Pepper", "Potato",
        "Tomato", "Apple", "Blueberry", "Cherry", "Corn", "Grape",
        "Orange", "Peach", "Raspberry", "Soybean", "Squash", "Strawberry"
    ]

    @classmethod
    def parse_class_name(cls, raw_class: str) -> Dict[str, Any]:
        """
        Parses raw PlantVillage class format into structured information.
        Handles 'Tomato___Late_blight', 'Tomato__Target_Spot', 'Tomato_Leaf_Mold', etc.
        """
        crop_part = "Plant"
        disease_part = raw_class

        if "___" in raw_class:
            crop_part, disease_part = raw_class.split("___", 1)
        elif "__" in raw_class:
            crop_part, disease_part = raw_class.split("__", 1)
        else:
            # Check if raw_class starts with any known crop name followed by '_'
            for crop_name in cls.KNOWN_CROPS:
                clean_c = crop_name.replace(",", "").replace("_", " ").strip()
                if raw_class.lower().startswith(crop_name.lower() + "_"):
                    crop_part = crop_name
                    disease_part = raw_class[len(crop_name) + 1:]
                    break
                elif raw_class.lower().startswith(clean_c.lower().replace(" ", "_") + "_"):
                    crop_part = clean_c
                    disease_part = raw_class[len(clean_c.replace(" ", "_")) + 1:]
                    break

        crop = re.sub(r'[\s_,]+', ' ', crop_part).strip().title()
        disease_clean = re.sub(r'[\s_]+', ' ', disease_part).strip().title()
        is_healthy = "healthy" in disease_part.lower()

        # Find matching advice description by key or substring
        advice = "Maintain optimal watering, inspect surrounding crops, and isolate any severely damaged plants."
        for key, text in cls.DISEASE_DESCRIPTIONS.items():
            key_clean = key.replace("_", " ").lower()
            if key.lower() in disease_part.lower() or key_clean in disease_clean.lower():
                advice = text
                break

        return {
            "raw_class": raw_class,
            "crop": crop,
            "condition": disease_clean,
            "display_name": f"{crop}: {disease_clean}",
            "is_healthy": is_healthy,
            "advice": advice,
        }

    def predict(
        self,
        image_input: Any,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Runs inference on the provided leaf image crop.

        Args:
            image_input: PIL Image or numpy array (RGB).
            top_k: Number of highest probability classes to return.

        Returns:
            Dict containing predicted top-1 class details and top-k distributions.
        """
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input.astype("uint8"))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Transform and forward pass
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # Top-k predictions
        top_k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top_1_idx = int(top_indices[0])
        top_1_raw = self.idx_to_class.get(top_1_idx, f"Class_{top_1_idx}")
        top_1_conf = float(top_probs[0])
        parsed = self.parse_class_name(top_1_raw)

        top_k_list = []
        for prob, idx in zip(top_probs, top_indices):
            raw_name = self.idx_to_class.get(int(idx), f"Class_{idx}")
            p_meta = self.parse_class_name(raw_name)
            top_k_list.append({
                "class_index": int(idx),
                "raw_name": raw_name,
                "display_name": p_meta["display_name"],
                "probability": float(prob),
                "is_healthy": p_meta["is_healthy"],
            })

        return {
            "class_index": top_1_idx,
            "raw_class": top_1_raw,
            "display_name": parsed["display_name"],
            "crop": parsed["crop"],
            "condition": parsed["condition"],
            "is_healthy": parsed["is_healthy"],
            "confidence": top_1_conf,
            "advice": parsed["advice"],
            "top_k": top_k_list,
            # Callers MUST check this before presenting any of the fields
            # above as a diagnosis. False means the classification head is
            # untrained and every field above is meaningless noise.
            "weights_loaded": self.weights_loaded,
        }
