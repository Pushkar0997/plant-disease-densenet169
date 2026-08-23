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
# HF_REPO_ID is intentionally an obvious placeholder rather than a guessed
# real repo: pointing it at a real-looking but wrong/nonexistent repo by
# default would fail in a way that's easy to miss (404 looks the same as
# "not configured yet"). Set PLANT_DISEASE_HF_REPO_ID once the checkpoint is
# actually uploaded to the Hub.
DEFAULT_HF_REPO_ID = "CHANGE_ME/plant-disease-densenet169"
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
            back to DEFAULT_HF_REPO_ID (an obvious placeholder — see above).
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

    if not repo_id or repo_id == DEFAULT_HF_REPO_ID:
        return WeightsResolution(
            path=None,
            source="none",
            error=(
                f"No local checkpoint at {local_path!r} and no Hugging Face "
                "repo is configured (PLANT_DISEASE_HF_REPO_ID is unset — it "
                f"still points at the placeholder {DEFAULT_HF_REPO_ID!r}). "
                "Set PLANT_DISEASE_HF_REPO_ID to the real repo once the "
                "checkpoint is uploaded, or place the .pth file at "
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
