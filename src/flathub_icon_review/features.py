"""Image-only features: a cheap baseline and optional pretrained CLIP embeddings."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps

BACKGROUNDS = ((250, 250, 250), (36, 36, 36))
FEATURE_VERSION = 1
CLIP_MODEL = "openai/clip-vit-base-patch32"


def views(path: Path) -> list[Image.Image]:
    with Image.open(path) as original:
        rgba = original.convert("RGBA")
    # Pad instead of cropping away footprint/margins.
    rgba = ImageOps.pad(rgba, (224, 224), color=(0, 0, 0, 0))
    return [
        Image.alpha_composite(Image.new("RGBA", rgba.size, (*bg, 255)), rgba).convert(
            "RGB"
        )
        for bg in BACKGROUNDS
    ]


def handcrafted(path: Path) -> np.ndarray:
    with Image.open(path) as original:
        rgba = np.asarray(original.convert("RGBA").resize((64, 64))) / 255.0
    alpha = rgba[:, :, 3]
    visible = alpha > 0.05
    coords = np.argwhere(visible)
    bounds = (
        [*coords.min(axis=0) / 64, *(coords.max(axis=0) + 1) / 64]
        if coords.size
        else [0.0] * 4
    )
    features = [
        alpha.mean(),
        visible.mean(),
        ((alpha > 0) & (alpha < 1)).mean(),
        *bounds,
    ]
    features.extend(
        np.asarray(
            Image.fromarray((alpha * 255).astype("uint8")).resize((8, 8))
        ).flatten()
        / 255
    )
    for image in views(path):
        rgb = np.asarray(image.resize((32, 32))) / 255.0
        gray = rgb.mean(axis=2)
        features.extend(rgb.mean(axis=(0, 1)))
        features.extend(rgb.std(axis=(0, 1)))
        for channel in range(3):
            hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0, 1))
            features.extend(hist / hist.sum())
        features.extend(
            [np.abs(np.diff(gray, axis=0)).mean(), np.abs(np.diff(gray, axis=1)).mean()]
        )
        features.extend(np.asarray(image.resize((8, 8))).flatten() / 255)
    return np.asarray(features, dtype=np.float32)


class Extractor:
    def __init__(self, kind="handcrafted", revision=None):
        self.kind = kind
        self.spec = {"kind": kind, "version": FEATURE_VERSION}
        if kind == "clip":
            import torch
            from transformers import CLIPModel, CLIPProcessor

            torch.set_num_threads(min(4, torch.get_num_threads()))
            self.model = CLIPModel.from_pretrained(CLIP_MODEL, revision=revision).eval()
            resolved = self.model.config._commit_hash
            self.processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL, revision=resolved or revision, use_fast=False
            )
            self.spec.update(model=CLIP_MODEL, revision=resolved or revision)
        elif kind != "handcrafted":
            raise ValueError(f"Unknown feature type: {kind}")

    def extract(self, path: Path) -> np.ndarray:
        if self.kind == "handcrafted":
            return handcrafted(path)
        import torch

        inputs = self.processor(images=views(path), return_tensors="pt")
        with torch.inference_mode():
            vectors = self.model.get_image_features(**inputs)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.cpu().numpy().flatten()

    def cached(self, root: Path, record: dict) -> np.ndarray:
        key = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()
        path = root / "features" / key / f"{record['icon_sha256']}.npy"
        if path.exists():
            try:
                return np.load(path, allow_pickle=False)
            except (ValueError, EOFError, OSError):
                # A cache is disposable; recover interrupted or damaged writes.
                pass
        result = self.extract(root / record["icon_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as file:
                temporary = Path(file.name)
                np.save(file, result)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return result
