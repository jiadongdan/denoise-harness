"""Portable first-run configuration for an installed denoise-learn provider."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from .config import CONFIG_SCHEMA_VERSION
from .provider import PROVIDER_CONTRACT_VERSION


def _resolve_python(value: str | Path | None) -> Path:
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return Path(sys.executable).resolve()
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    discovered = shutil.which(str(value))
    if discovered is not None:
        return Path(discovered).resolve()
    return candidate.resolve()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_assignments(values: Iterable[str], label: str) -> dict[str, str]:
    """Parse repeated METHOD=VALUE command-line assignments."""
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use METHOD=VALUE syntax: {value!r}")
        identifier, assigned = value.split("=", 1)
        identifier = identifier.strip()
        assigned = assigned.strip()
        if not identifier or not assigned:
            raise ValueError(f"{label} must use non-empty METHOD=VALUE values.")
        if identifier in assignments:
            raise ValueError(f"Duplicate {label} assignment for {identifier}.")
        assignments[identifier] = assigned
    return assignments


def discover_provider(python_executable: str | Path | None = None) -> dict[str, Any]:
    """Query the published provider through the selected Python interpreter."""
    executable = _resolve_python(python_executable)
    completed = subprocess.run(
        [
            str(executable),
            "-m",
            "denoiselearn.provider.worker",
            "--capabilities",
        ],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "A compatible denoise-learn provider was not discovered in "
            f"{executable}. Install it with: \"{executable}\" -m pip install "
            f"\"denoise-learn[inference]\". Provider detail: {detail}"
        )
    try:
        capabilities = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"denoise-learn returned invalid capability JSON: {error}") from error
    if capabilities.get("contract_version") != PROVIDER_CONTRACT_VERSION:
        raise RuntimeError(
            "The installed denoise-learn provider uses an incompatible contract."
        )
    return capabilities


def _method_entry(
    capability: dict[str, Any],
    provider_version: str,
    provider_python: Path,
    checkpoint_path: Path | None,
    state_key: str | None,
    schema_version: str | None,
    checkpoint_model_name: str | None,
) -> dict[str, Any]:
    identifier = str(capability["identifier"])
    entry: dict[str, Any] = {
        "identifier": identifier,
        "kind": str(capability["kind"]),
        "family": str(capability.get("family", capability["kind"])),
        "implementation_version": provider_version,
        "python_executable": str(provider_python),
        "device_kind": str(capability.get("device_kind", "cpu")),
        "internal_input_range": capability.get("internal_input_range", [0.0, 1.0]),
        "raw_output_range": str(capability.get("raw_output_range", "unbounded")),
        "size_multiple": capability.get("size_multiple", 1),
        "postprocess": str(capability.get("postprocess", "native_unit")),
        "defaults": dict(capability.get("defaults", {})),
        "parameter_schema": dict(capability.get("parameter_schema", {})),
        "reference": f"denoise-learn {provider_version} provider capability",
    }
    model_name = capability.get("model_name")
    if model_name is not None:
        entry["model_name"] = str(model_name)
    if checkpoint_path is not None:
        entry["checkpoint_path"] = str(checkpoint_path.resolve())
        entry["checkpoint_sha256"] = _file_sha256(checkpoint_path)
        adapter_options: dict[str, str] = {}
        if state_key is not None:
            adapter_options["checkpoint_state_key"] = state_key
        if schema_version is not None:
            adapter_options["checkpoint_schema_version"] = schema_version
        if checkpoint_model_name is not None:
            adapter_options["checkpoint_model_name"] = checkpoint_model_name
        entry["adapter_options"] = adapter_options
    return entry


def initialize_config(
    output_path: str | Path,
    *,
    provider_python: str | Path | None = None,
    checkpoints: dict[str, str] | None = None,
    state_keys: dict[str, str] | None = None,
    schema_versions: dict[str, str] | None = None,
    checkpoint_model_names: dict[str, str] | None = None,
    force: bool = False,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover methods, write a portable config, and report checkpoint next steps."""
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(f"Config already exists: {output}. Use --force to replace it.")
    provider_python_path = _resolve_python(provider_python)
    discovered = capabilities or discover_provider(provider_python_path)
    checkpoint_values = checkpoints or {}
    state_key_values = state_keys or {}
    schema_values = schema_versions or {}
    model_name_values = checkpoint_model_names or {}
    capability_methods = list(discovered.get("methods", []))
    known = {str(method.get("identifier")) for method in capability_methods}
    supplied_identifiers = (
        set(checkpoint_values)
        | set(state_key_values)
        | set(schema_values)
        | set(model_name_values)
    )
    unknown = sorted(supplied_identifiers.difference(known))
    if unknown:
        raise ValueError(f"Checkpoint supplied for unknown methods: {', '.join(unknown)}")
    methods: list[dict[str, Any]] = []
    guidance: list[dict[str, str]] = []
    provider_version = str(discovered.get("provider_version", "")).strip()
    if not provider_version:
        raise RuntimeError("Provider discovery did not report a provider version.")
    for capability in capability_methods:
        identifier = str(capability["identifier"])
        if not bool(capability.get("available", True)):
            guidance.append(
                {
                    "identifier": identifier,
                    "status": "dependency_unavailable",
                    "action": str(
                        capability.get(
                            "unavailable_reason",
                            "Install this method's optional denoise-learn dependencies.",
                        )
                    ),
                }
            )
            continue
        requires_checkpoint = bool(capability.get("checkpoint_required", False))
        checkpoint = None
        if identifier in checkpoint_values:
            checkpoint = Path(checkpoint_values[identifier]).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist for {identifier}: {checkpoint}")
        if requires_checkpoint and checkpoint is None:
            guidance.append(
                {
                    "identifier": identifier,
                    "status": "not_configured",
                    "action": (
                        "Run denoise init --force with --checkpoint "
                        f"{identifier}=PATH. Add --checkpoint-state-key, "
                        "--checkpoint-schema-version, or --checkpoint-model-name "
                        "when the checkpoint is a wrapped training artifact."
                    ),
                }
            )
            continue
        methods.append(
            _method_entry(
                capability,
                provider_version,
                provider_python_path,
                checkpoint,
                state_key_values.get(identifier),
                schema_values.get(identifier),
                model_name_values.get(identifier),
            )
        )
    if not methods:
        raise RuntimeError("Provider discovery produced no immediately configurable methods.")
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "project_root": ".",
        "output_root": "denoise-runs",
        "provider": {
            "name": str(discovered.get("provider", "denoise-learn")),
            "module": "denoiselearn.provider.worker",
            "distribution": "denoise-learn",
            "minimum_version": provider_version,
        },
        "defaults": {
            "comparison_policy": "clipped_0_1",
            "input_normalization": "minmax_0_1",
            "reference_normalization": "minmax_0_1",
            "search_budget": 6,
            "timeout_seconds": 900,
        },
        "methods": methods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "configured",
        "config_path": str(output),
        "python_executable": str(provider_python_path),
        "provider": str(discovered.get("provider", "denoise-learn")),
        "provider_version": provider_version,
        "enabled_methods": [method["identifier"] for method in methods],
        "checkpoint_guidance": guidance,
        "next_command": f'denoise doctor --config "{output}"',
    }
