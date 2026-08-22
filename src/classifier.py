"""
Plant Disease Classifier using DenseNet169 Transfer Learning.
Supports PyTorch model creation, weights loading, preprocessing, and top-k inference.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
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

        # Build DenseNet169 model architecture
        self.model = self._build_model(self.num_classes)

        # Load weights if provided
        self.weights_loaded = False
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            if weights_path:
                print(f"[DiseaseClassifier] Weights file not found at {weights_path}. Running with pre-configured weights.")

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

    def load_weights(self, weights_path: str):
        """Loads state dict into model."""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
            self.weights_loaded = True
            print(f"[DiseaseClassifier] Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"[DiseaseClassifier] Failed to load checkpoint {weights_path}: {e}")

    @staticmethod
    def parse_class_name(raw_class: str) -> Dict[str, Any]:
        """
        Parses raw PlantVillage class format into structured information.
        Example: 'Tomato___Late_blight' -> Crop: Tomato, Disease: Late Blight, is_healthy: False
        """
        if "___" in raw_class:
            crop_part, disease_part = raw_class.split("___", 1)
        else:
            crop_part, disease_part = "Plant", raw_class

        crop = crop_part.replace("_", " ").strip()
        disease_clean = disease_part.replace("_", " ").strip()
        is_healthy = "healthy" in disease_part.lower()

        # Find matching advice description
        advice = DiseaseClassifier.DISEASE_DESCRIPTIONS.get(
            disease_part,
            "Maintain optimal watering, inspect surrounding crops, and isolate any severely damaged plants."
        )

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
        }
