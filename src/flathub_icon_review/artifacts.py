"""Trusted local model loading and atomic artifact writes.

Joblib uses pickle: validation is not a sandbox. Never load untrusted models.
"""

from pathlib import Path
from pickle import UnpicklingError
from struct import error as StructError
from tempfile import NamedTemporaryFile

import joblib


def atomic_dump(value, path: Path):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            joblib.dump(value, file)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_model(path: Path, kind: str):
    try:
        artifact = joblib.load(path)
    except (EOFError, KeyError, IndexError, UnpicklingError, StructError) as error:
        raise ValueError(
            f"Unreadable model artifact: {path}; download it again"
        ) from error
    if not isinstance(artifact, dict) or not isinstance(artifact.get("models"), dict):
        raise ValueError(f"Expected an {kind} model artifact: {path}")
    if kind == "icon":
        spec = artifact.get("features")
        if not isinstance(spec, dict) or spec.get("kind") not in (
            "handcrafted",
            "clip",
        ):
            raise ValueError("Expected an icon model trained with 'appraise train'")
    elif artifact.get("artifact_type") != "listing":
        raise ValueError("Unsupported listing model; train with train-listing")
    calibrators = artifact.get("calibrators", {})
    if not isinstance(calibrators, dict):
        raise ValueError("Invalid calibration models in artifact")
    for key, model in artifact["models"].items():
        if not isinstance(key, str) or not callable(
            getattr(model, "predict_proba", None)
        ):
            raise ValueError("Invalid classifier in model artifact")
        if key in calibrators and (
            not callable(getattr(model, "decision_function", None))
            or not callable(getattr(calibrators[key], "predict_proba", None))
        ):
            raise ValueError("Invalid calibrator in model artifact")
    return artifact
