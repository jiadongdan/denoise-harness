"""Load and validate portable denoise-harness method manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from .contracts import MethodMetadata


CONFIG_SCHEMA_VERSION = "scientific-denoise-harness-config-v1"
SUPPORTED_METHOD_KINDS = {"fft", "svd", "asn"}


@dataclass(frozen=True)
class ProviderConfig:
    """External denoise execution provider used by the Harness."""

    name: str
    module: str
    distribution: str
    minimum_version: str | None
    source_root: Path | None
    install_hint: str | None


@dataclass(frozen=True)
class MethodConfig:
    """Resolved method definition from a portable JSON manifest."""

    metadata: MethodMetadata
    kind: str
    model_name: str | None
    postprocess: str
    defaults: dict[str, Any]
    adapter_options: dict[str, Any]


@dataclass(frozen=True)
class HarnessConfig:
    """Validated Harness configuration with resolved absolute paths."""

    source_path: Path
    output_root: Path
    project_root: Path
    methods: dict[str, MethodConfig]
    defaults: dict[str, Any]
    provider: ProviderConfig


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_python(base: Path, value: str | None) -> Path:
    """Resolve a configured interpreter, including the portable auto sentinel."""
    requested = "auto" if value is None else str(value).strip()
    if requested in {"", "auto"}:
        return Path(sys.executable).resolve()
    path = Path(requested).expanduser()
    if path.is_absolute():
        return path.resolve()
    discovered = shutil.which(requested)
    if discovered is not None:
        return Path(discovered).resolve()
    return (base / path).resolve()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return dict(value)


def load_harness_config(path: str | Path) -> HarnessConfig:
    """Load one explicit config without searching project-specific locations."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Agent config does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported denoise-harness config schema version.")
    config_dir = source.parent
    project_root = _resolve_path(config_dir, str(payload.get("project_root", ".")))
    assert project_root is not None
    output_root = _resolve_path(project_root, str(payload.get("output_root", "denoise-runs")))
    assert output_root is not None
    defaults = _require_mapping(payload.get("defaults", {}), "defaults")
    provider_payload = _require_mapping(payload.get("provider", {}), "provider")
    provider_source_root = _resolve_path(
        config_dir, provider_payload.get("source_root")
    )
    provider = ProviderConfig(
        name=str(provider_payload.get("name", "denoise-learn")),
        module=str(
            provider_payload.get("module", "denoiselearn.provider.worker")
        ),
        distribution=str(
            provider_payload.get("distribution", "denoise-learn")
        ),
        minimum_version=(
            None
            if provider_payload.get("minimum_version") is None
            else str(provider_payload["minimum_version"])
        ),
        source_root=provider_source_root,
        install_hint=(
            None
            if provider_payload.get("install_hint") is None
            else str(provider_payload["install_hint"])
        ),
    )
    method_entries = payload.get("methods")
    if not isinstance(method_entries, list) or not method_entries:
        raise ValueError("Config must contain at least one method entry.")
    methods: dict[str, MethodConfig] = {}
    for raw_entry in method_entries:
        entry = _require_mapping(raw_entry, "method entry")
        if entry.get("enabled", True) is False:
            continue
        identifier = str(entry.get("identifier", "")).strip()
        if not identifier or identifier in methods:
            raise ValueError(f"Invalid or duplicate method identifier: {identifier!r}")
        kind = str(entry.get("kind", ""))
        if kind not in SUPPORTED_METHOD_KINDS:
            raise ValueError(f"Unsupported method kind for {identifier}: {kind}")
        python_path = _resolve_python(project_root, entry.get("python_executable"))
        checkpoint_path = _resolve_path(project_root, entry.get("checkpoint_path"))
        if kind == "asn" and checkpoint_path is None:
            raise ValueError(f"Checkpoint path is missing for {identifier}.")
        checkpoint_sha256 = entry.get("checkpoint_sha256")
        if checkpoint_sha256 is not None:
            checkpoint_sha256 = str(checkpoint_sha256).lower()
            if len(checkpoint_sha256) != 64:
                raise ValueError(f"Invalid checkpoint SHA-256 for {identifier}.")
        internal_range = entry.get("internal_input_range", [0.0, 1.0])
        if internal_range is not None:
            if not isinstance(internal_range, list) or len(internal_range) != 2:
                raise ValueError(f"Invalid internal range for {identifier}.")
            internal_range = (float(internal_range[0]), float(internal_range[1]))
        size_multiple = entry.get("size_multiple")
        metadata = MethodMetadata(
            identifier=identifier,
            family=str(entry.get("family", kind)),
            implementation_version=str(entry.get("implementation_version", "unknown")),
            device_kind=str(entry.get("device_kind", "cpu")),
            python_executable=str(python_path),
            internal_input_range=internal_range,
            raw_output_range=str(entry.get("raw_output_range", "unbounded")),
            size_multiple=None if size_multiple is None else int(size_multiple),
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            reference=entry.get("reference"),
            parameter_schema=_require_mapping(entry.get("parameter_schema", {}), "parameter_schema"),
        )
        methods[identifier] = MethodConfig(
            metadata=metadata,
            kind=kind,
            model_name=entry.get("model_name"),
            postprocess=str(entry.get("postprocess", "clipped_0_1")),
            defaults=_require_mapping(entry.get("defaults", {}), "method defaults"),
            adapter_options=_require_mapping(entry.get("adapter_options", {}), "adapter_options"),
        )
    if not methods:
        raise ValueError("Config must enable at least one method entry.")
    return HarnessConfig(
        source_path=source,
        output_root=output_root,
        project_root=project_root,
        methods=methods,
        defaults=defaults,
        provider=provider,
    )
