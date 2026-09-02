"""Deterministic candidate generation for tunable classical denoisers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import MethodConfig
from .diagnostics import estimate_noise_sigma


def _schema_candidates(method: MethodConfig, name: str) -> list[Any]:
    specification = method.metadata.parameter_schema.get(name, {})
    if not isinstance(specification, dict):
        raise TypeError(f"Parameter schema for {name} must be an object.")
    candidates = specification.get("candidates", [])
    if not isinstance(candidates, list):
        raise TypeError(f"Parameter candidates for {name} must be a list.")
    return list(candidates)


def fft_candidates(
    method: MethodConfig, image: np.ndarray, budget: int
) -> list[dict[str, Any]]:
    """Order FFT keep fractions by a reproducible noise-level prior."""
    values = [float(value) for value in _schema_candidates(method, "p")]
    if not values:
        values = [0.01, 0.02, 0.05, 0.1, 0.2]
    sigma = estimate_noise_sigma(image)
    preferred = 0.02 if sigma >= 0.12 else 0.05 if sigma >= 0.06 else 0.1
    ordered = sorted(set(values), key=lambda value: (abs(np.log(value / preferred)), value))
    return [{"p": value} for value in ordered[:budget]]


def svd_candidates(
    method: MethodConfig, image: np.ndarray, budget: int
) -> list[dict[str, Any]]:
    """Generate legal SVD patch/component pairs from the method manifest."""
    schema_pairs = method.metadata.parameter_schema.get("candidate_pairs", [])
    if schema_pairs and not isinstance(schema_pairs, list):
        raise TypeError("SVD candidate_pairs must be a list.")
    if schema_pairs:
        pairs = [dict(pair) for pair in schema_pairs]
    else:
        pairs = [
            {"patch_size": 8, "n_components": 4},
            {"patch_size": 8, "n_components": 8},
            {"patch_size": 16, "n_components": 8},
            {"patch_size": 16, "n_components": 16},
            {"patch_size": 32, "n_components": 16},
            {"patch_size": 32, "n_components": 32},
            {"patch_size": 32, "n_components": 64},
            {"patch_size": 64, "n_components": 64},
        ]
    minimum_dimension = min(image.shape)
    legal: list[dict[str, Any]] = []
    for pair in pairs:
        patch_size = int(pair["patch_size"])
        components = int(pair["n_components"])
        if 2 <= patch_size < minimum_dimension and 1 <= components <= patch_size**2:
            legal.append(
                {
                    "patch_size": patch_size,
                    "n_components": components,
                    "random_seed": int(pair.get("random_seed", 0)),
                }
            )
    if not legal:
        raise ValueError("No legal SVD parameter candidates exist for the input shape.")
    sigma = estimate_noise_sigma(image)
    preferred_patch = 32 if minimum_dimension > 64 else 16 if minimum_dimension > 32 else 8
    preferred_components = 16 if sigma >= 0.08 else 32
    legal.sort(
        key=lambda item: (
            abs(np.log(item["patch_size"] / preferred_patch))
            + 0.5 * abs(np.log(item["n_components"] / preferred_components)),
            item["patch_size"],
            item["n_components"],
        )
    )
    return legal[:budget]


def generate_candidates(
    method: MethodConfig, image: np.ndarray, budget: int
) -> list[dict[str, Any]]:
    """Return deterministic candidates for one supported tunable method."""
    if budget < 1:
        raise ValueError("Search budget must be positive.")
    if method.kind == "mtflearn_fft":
        return fft_candidates(method, image, budget)
    if method.kind == "mtflearn_svd":
        return svd_candidates(method, image, budget)
    raise ValueError(f"Method does not support parameter search: {method.metadata.identifier}")
