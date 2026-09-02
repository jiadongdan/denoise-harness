"""Reference metrics and conservative no-reference denoising diagnostics."""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def reference_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute unit-range full-image metrics against an explicit clean reference."""
    clean = np.asarray(reference, dtype=np.float32)
    output = np.asarray(prediction, dtype=np.float32)
    if clean.shape != output.shape or clean.ndim != 2:
        raise ValueError("Reference and prediction must be equal-shape 2D arrays.")
    if not np.isfinite(clean).all() or not np.isfinite(output).all():
        raise ValueError("Metric inputs must be finite.")
    error = output - clean
    mse = float(np.mean(np.square(error), dtype=np.float64))
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))
    ssim = float(structural_similarity(clean, output, data_range=1.0))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def _gradient_rms(image: np.ndarray) -> float:
    dy = np.diff(image, axis=0)
    dx = np.diff(image, axis=1)
    return float(np.sqrt((np.mean(dx * dx) + np.mean(dy * dy)) / 2.0))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate white-noise scale from a robust diagonal high-pass response."""
    values = np.asarray(image, dtype=np.float32)
    if min(values.shape) < 3:
        return 0.0
    response = (
        values[:-2, :-2]
        - values[:-2, 2:]
        - values[2:, :-2]
        + values[2:, 2:]
    )
    median = float(np.median(response))
    mad = float(np.median(np.abs(response - median)))
    return mad / (0.67448975 * 2.0) if mad > 0.0 else 0.0


def _periodic_peak_ratio(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.float64)
    centered = values - values.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    height, width = spectrum.shape
    radius = max(1, min(height, width) // 32)
    spectrum[
        max(0, height // 2 - radius) : min(height, height // 2 + radius + 1),
        max(0, width // 2 - radius) : min(width, width // 2 + radius + 1),
    ] = 0.0
    positive = spectrum[spectrum > 0.0]
    if positive.size < 10:
        return 0.0
    return float(np.quantile(positive, 0.999) / max(np.median(positive), 1e-12))


def no_reference_diagnostics(
    input_image: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Return descriptive diagnostics that do not claim ground-truth quality."""
    source = np.asarray(input_image, dtype=np.float32)
    output = np.asarray(prediction, dtype=np.float32)
    if source.shape != output.shape or source.ndim != 2:
        raise ValueError("Input and prediction must be equal-shape 2D arrays.")
    residual = source - output
    input_gradient = _gradient_rms(source)
    output_gradient = _gradient_rms(output)
    residual_vertical = _safe_correlation(residual[:-1], residual[1:])
    residual_horizontal = _safe_correlation(residual[:, :-1], residual[:, 1:])
    return {
        "input_noise_sigma_estimate": estimate_noise_sigma(source),
        "residual_standard_deviation": float(np.std(residual, dtype=np.float64)),
        "residual_input_correlation": _safe_correlation(residual, source),
        "residual_neighbor_correlation_abs_mean": float(
            (abs(residual_vertical) + abs(residual_horizontal)) / 2.0
        ),
        "gradient_rms_ratio": float(output_gradient / max(input_gradient, 1e-12)),
        "residual_periodic_peak_ratio": _periodic_peak_ratio(residual),
    }


def heuristic_candidate_score(diagnostics: dict[str, float]) -> float:
    """Rank no-reference ML candidates with an explicit conservative v1 heuristic."""
    sigma = max(float(diagnostics["input_noise_sigma_estimate"]), 1e-6)
    removal_ratio = float(diagnostics["residual_standard_deviation"]) / sigma
    gradient_ratio = float(diagnostics["gradient_rms_ratio"])
    removal_penalty = abs(math.log(max(removal_ratio, 1e-6)))
    structure_penalty = max(0.0, 0.55 - gradient_ratio) * 4.0
    input_leakage = abs(float(diagnostics["residual_input_correlation"])) * 1.5
    neighbor_penalty = float(diagnostics["residual_neighbor_correlation_abs_mean"])
    periodic_penalty = math.log1p(float(diagnostics["residual_periodic_peak_ratio"])) * 0.05
    return float(
        -(removal_penalty + structure_penalty + input_leakage + neighbor_penalty + periodic_penalty)
    )
