"""Input loading, normalization, checksums, and display artifact helpers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".npy", ".png", ".tif", ".tiff"}
NORMALIZATION_POLICIES = {"minmax_0_1", "already_0_1", "dtype_unit"}


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_single_channel(path: str | Path) -> np.ndarray:
    """Load one numeric 2D image without applying hidden normalization."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {suffix}")
    if suffix == ".npy":
        image = np.load(source, allow_pickle=False)
    else:
        with Image.open(source) as handle:
            image = np.asarray(handle)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected one 2D channel, received shape {image.shape}.")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Expected a numeric image, received dtype {image.dtype}.")
    if not np.isfinite(image).all():
        raise ValueError("Input image contains NaN or infinite values.")
    return image


def normalize_image(image: np.ndarray, policy: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a numeric image into the explicit shared float32 unit space."""
    if policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"Unsupported normalization policy: {policy}")
    original = np.asarray(image)
    values = original.astype(np.float32, copy=False)
    minimum = float(values.min())
    maximum = float(values.max())
    if policy == "minmax_0_1":
        if maximum == minimum:
            raise ValueError("A constant image cannot use min-max normalization.")
        unit = (values - minimum) / (maximum - minimum)
        parameters = {"minimum": minimum, "maximum": maximum}
    elif policy == "already_0_1":
        tolerance = 1e-6
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            raise ValueError("already_0_1 input lies outside [0, 1].")
        unit = values.copy()
        parameters = {}
    else:
        if np.issubdtype(original.dtype, np.integer):
            dtype_info = np.iinfo(original.dtype)
            if dtype_info.min < 0:
                raise ValueError("dtype_unit does not support signed integer inputs.")
            unit = values / float(dtype_info.max)
            parameters = {"dtype_maximum": int(dtype_info.max)}
        else:
            tolerance = 1e-6
            if minimum < -tolerance or maximum > 1.0 + tolerance:
                raise ValueError("Floating dtype_unit input lies outside [0, 1].")
            unit = values.copy()
            parameters = {}
    unit = np.asarray(unit, dtype=np.float32)
    if not np.isfinite(unit).all() or float(unit.min()) < -1e-5 or float(unit.max()) > 1.00001:
        raise RuntimeError("Normalization did not produce a finite unit-range image.")
    return unit, {
        "policy": policy,
        "original_dtype": str(original.dtype),
        "original_range": [minimum, maximum],
        "output_dtype": "float32",
        "output_range": [float(unit.min()), float(unit.max())],
        "parameters": parameters,
    }


def inspect_input(path: str | Path, normalization_policy: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, normalize, and describe one input image."""
    source = Path(path).expanduser().resolve()
    raw = load_single_channel(source)
    unit, transform = normalize_image(raw, normalization_policy)
    record = {
        "path": str(source),
        "sha256": file_sha256(source),
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "finite": True,
        "normalization": transform,
    }
    return unit, record


def save_preview(path: str | Path, image: np.ndarray) -> None:
    """Save one clipped unit-range array as an 8-bit PNG preview."""
    values = np.asarray(image, dtype=np.float32)
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels).save(Path(path))
