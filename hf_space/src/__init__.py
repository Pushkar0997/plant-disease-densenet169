"""
Plant Disease Detection & Visual Diagnostic Pipeline
Modular package integrating leaf localization with fine-tuned DenseNet169 transfer learning.
"""

from .detector import LeafDetector
from .classifier import DiseaseClassifier
from .visualizer import Visualizer
from .pipeline import PlantDiagnosticPipeline
from .weights import resolve_weights_path, WeightsResolution

__version__ = "1.0.0"
__all__ = [
    "LeafDetector",
    "DiseaseClassifier",
    "Visualizer",
    "PlantDiagnosticPipeline",
    "resolve_weights_path",
    "WeightsResolution",
]
