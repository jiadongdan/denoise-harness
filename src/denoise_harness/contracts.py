"""Versioned data contracts shared by the denoise harness and its workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CONTRACT_VERSION = "scientific-denoise-harness-v1"


@dataclass(frozen=True)
class MethodMetadata:
    """Provenance and numeric constraints for one denoising method."""

    identifier: str
    family: str
    implementation_version: str
    device_kind: str
    python_executable: str
    internal_input_range: tuple[float, float] | None
    raw_output_range: str
    size_multiple: int | None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    reference: str | None = None
    parameter_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunPlan:
    """Validated execution plan produced before any method is run."""

    methods: tuple[str, ...]
    method_options: dict[str, dict[str, Any]]
    auto_ml_parameters: bool
    search_budget: int
    comparison_policy: str
    input_normalization: str
    reference_normalization: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DenoiseResult:
    """Artifact-oriented result returned by one deterministic method call."""

    identifier: str
    metadata: MethodMetadata
    options: dict[str, Any]
    raw_output_path: str
    comparison_output_path: str
    residual_path: str
    runtime_seconds: float
    raw_output_range: tuple[float, float]
    comparison_output_range: tuple[float, float]
    transforms: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metrics: dict[str, float]
    warnings: list[str]
    worker_record_path: str
    search: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["contract_version"] = CONTRACT_VERSION
        return result
