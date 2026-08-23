"""
Checkpoint resolution: local file first, Hugging Face Hub fallback.

This module answers exactly one question — "what local .pth path (if any)
should DiseaseClassifier try to load?" — and never itself decides that a
missing/failed checkpoint is okay to paper over. If nothing usable is found,
it returns None plus a clear reason string; it is the caller's job (app.py)
to make sure that reason ends up in front of the user, and DiseaseClassifier's
job (see src/classifier.py) to make sure a None path never gets treated as a
trained model.
"""

import os
from typing import NamedTuple, Optional

# Sensible defaults, both overridable via environment variables so a
# deployment (e.g. a Hugging Face Space) doesn't require code changes.
#
# DEFAULT_HF_REPO_ID points at the published checkpoint, so a fresh clone
# with no local .pth fetches it automatically and the app works out of the
# box. Override with PLANT_DISEASE_HF_REPO_ID to use your own fine-tune.
DEFAULT_HF_REPO_ID = "PushkarKumar/plant-disease-densenet169"
DEFAULT_HF_FILENAME = "densenet169_plant_disease.pth"


class WeightsResolution(NamedTuple):
    path: Optional[str]       # Local filesystem path to a checkpoint, or None.
    source: str                # "local", "hf_hub", or "none".
    error: Optional[str]       # Set whenever an *attempted* source failed.


def resolve_weights_path(
    local_path: str,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> WeightsResolution:
    """
    Resolve a checkpoint path, local file first, then the Hugging Face Hub.

    Args:
        local_path: Path checked first (e.g. models/densenet169_plant_disease.pth).
            If it exists, it is used and the Hub is never contacted — this is
            what keeps a local dev loop network-free.
        repo_id: Hugging Face repo id to download from if local_path is
            absent. Defaults to the PLANT_DISEASE_HF_REPO_ID env var, falling
            back to DEFAULT_HF_REPO_ID (the published checkpoint repo).
        filename: Filename within the repo. Defaults to the
            PLANT_DISEASE_HF_FILENAME env var, falling back to
            DEFAULT_HF_FILENAME.
        cache_dir: Optional huggingface_hub cache directory override.

    Returns:
        WeightsResolution. `path` is None if and only if no checkpoint could
        be resolved from either source; in that case `error` explains why,
        and callers must not treat this as "fine, just use a random head" —
        DiseaseClassifier already refuses to do that (weights_loaded stays
        False), but the reason should still be surfaced to the user rather
        than silently swallowed.
    """
    if local_path and os.path.exists(local_path):
        return WeightsResolution(path=local_path, source="local", error=None)

    repo_id = repo_id or os.environ.get("PLANT_DISEASE_HF_REPO_ID", DEFAULT_HF_REPO_ID)
    filename = filename or os.environ.get("PLANT_DISEASE_HF_FILENAME", DEFAULT_HF_FILENAME)

    if not repo_id:
        return WeightsResolution(
            path=None,
            source="none",
            error=(
                f"No local checkpoint at {local_path!r} and "
                "PLANT_DISEASE_HF_REPO_ID was set to an empty value, so there "
                "is no Hugging Face repo to fetch from. Either unset it to use "
                f"the default ({DEFAULT_HF_REPO_ID!r}) or place a .pth file at "
                f"{local_path!r}."
            ),
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        return WeightsResolution(
            path=None,
            source="none",
            error=(
                "huggingface_hub is not installed, so the checkpoint could "
                f"not be fetched from {repo_id!r}. Run "
                "`pip install -r requirements.txt`. "
                f"(ImportError: {e})"
            ),
        )

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        return WeightsResolution(path=downloaded_path, source="hf_hub", error=None)
    except Exception as e:
        # Deliberately loud and specific. A download failure must never be
        # allowed to look like "everything's fine, here's an untrained model"
        # — the caller is expected to surface `error` to the user, and
        # DiseaseClassifier's own weights_loaded=False guarantee (see
        # src/classifier.py) is what actually prevents a silent random-head
        # fallback from being presented as a diagnosis.
        return WeightsResolution(
            path=None,
            source="none",
            error=f"Failed to download checkpoint from Hugging Face Hub repo {repo_id!r} (file {filename!r}): {e}",
        )
