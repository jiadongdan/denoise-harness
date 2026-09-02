"""Client helpers and readiness checks for an external denoise provider."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from .config import HarnessConfig, MethodConfig


WORKER_SCHEMA_VERSION = "scientific-denoise-worker-v1"
PROVIDER_CONTRACT_VERSION = "denoise-learn-provider-v1"
PROVIDER_REPOSITORY_URL = "https://github.com/jiadongdan/denoise-learn"
PROVIDER_INSTALL_REQUIREMENT = (
    "denoise-learn[inference] @ "
    "git+https://github.com/jiadongdan/denoise-learn.git@main"
)


def default_provider_install_command(python_executable: str) -> str:
    """Return the public GitHub installation command for denoise-learn."""
    return f'"{python_executable}" -m pip install "{PROVIDER_INSTALL_REQUIREMENT}"'


def _version_tuple(value: str) -> tuple[int, ...] | None:
    """Parse the numeric release portion used by the provider compatibility check."""
    parts = value.split("+", 1)[0].split("-", 1)[0].split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _version_is_older(actual: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) < minimum + (0,) * (
        width - len(minimum)
    )


def validate_provider_capabilities(
    config: HarnessConfig, capabilities: dict[str, Any]
) -> list[str]:
    """Return compatibility issues for one provider capability document."""
    issues: list[str] = []
    if capabilities.get("contract_version") != PROVIDER_CONTRACT_VERSION:
        issues.append("provider contract version mismatch")
    if capabilities.get("provider") != config.provider.name:
        issues.append("unexpected provider identity")
    minimum = config.provider.minimum_version
    actual = str(capabilities.get("provider_version", "unknown"))
    if minimum is not None:
        actual_tuple = _version_tuple(actual)
        minimum_tuple = _version_tuple(minimum)
        if actual_tuple is None:
            issues.append("provider version is unavailable")
        elif minimum_tuple is not None and _version_is_older(
            actual_tuple, minimum_tuple
        ):
            issues.append(
                f"provider version {actual} is older than required {minimum}"
            )
    return issues


def provider_environment(config: HarnessConfig) -> dict[str, str]:
    """Build a subprocess environment with an optional development provider root."""
    environment = os.environ.copy()
    entries: list[str] = []
    if config.provider.source_root is not None:
        entries.append(str(config.provider.source_root))
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    if entries:
        environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def provider_working_directory(config: HarnessConfig) -> Path:
    """Return a stable provider working directory without repository discovery."""
    if config.provider.source_root is not None:
        return config.provider.source_root
    return config.source_path.parent


def provider_command(
    config: HarnessConfig, python_executable: str, action: str, path: Path | None = None
) -> list[str]:
    """Build one provider module invocation."""
    command = [python_executable, "-m", config.provider.module, action]
    if path is not None:
        command.append(str(path))
    return command


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_command(config: HarnessConfig, python_executable: str) -> str:
    if config.provider.install_hint:
        return config.provider.install_hint.replace("{python}", python_executable)
    if config.provider.source_root is not None:
        return (
            f'"{python_executable}" -m pip install -e '
            f'"{config.provider.source_root}[inference]"'
        )
    return default_provider_install_command(python_executable)


def run_provider_array(
    config: HarnessConfig,
    method: MethodConfig,
    image: np.ndarray,
    options: dict[str, Any],
    comparison_policy: str,
    *,
    timeout_seconds: int = 120,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one configured provider method for an in-memory normalized image."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Provider input must be a finite two-dimensional image.")
    with tempfile.TemporaryDirectory(prefix="denoise-harness-provider-") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npy"
        output_path = directory / "output.npz"
        record_path = directory / "record.json"
        job_path = directory / "job.json"
        np.save(input_path, values)
        payload = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "record_path": str(record_path),
            "comparison_policy": comparison_policy,
            "options": options,
            "method": {
                "identifier": method.metadata.identifier,
                "kind": method.kind,
                "model_name": method.model_name,
                "postprocess": method.postprocess,
                "size_multiple": method.metadata.size_multiple,
                "device_kind": method.metadata.device_kind,
                "checkpoint_path": method.metadata.checkpoint_path,
                "checkpoint_sha256": method.metadata.checkpoint_sha256,
                "adapter_options": method.adapter_options,
            },
        }
        job_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        completed = subprocess.run(
            provider_command(
                config, method.metadata.python_executable, "--job", job_path
            ),
            cwd=provider_working_directory(config),
            env=provider_environment(config),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Provider failed for {method.metadata.identifier}: {detail}"
            )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
            raise RuntimeError("Provider result contract version mismatch.")
        if record.get("provider") != config.provider.name:
            raise RuntimeError("Provider result identity mismatch.")
        with np.load(output_path, allow_pickle=False) as arrays:
            raw = np.asarray(arrays["raw"], dtype=np.float32)
            comparison = np.asarray(arrays["comparison"], dtype=np.float32)
        return raw, comparison, record


def doctor(config: HarnessConfig) -> dict[str, Any]:
    """Check provider installations, method runtimes, checkpoints, and devices."""
    environments: dict[str, dict[str, Any]] = {}
    recommendations: list[str] = []
    for method in config.methods.values():
        python_executable = method.metadata.python_executable
        if python_executable in environments:
            continue
        executable_path = Path(python_executable)
        if not executable_path.is_file():
            environments[python_executable] = {
                "status": "blocked",
                "error": "Python executable does not exist.",
            }
            recommendations.append(
                f"Configure an existing Python executable for {python_executable}."
            )
            continue
        completed = subprocess.run(
            provider_command(config, python_executable, "--capabilities"),
            cwd=provider_working_directory(config),
            env=provider_environment(config),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            environments[python_executable] = {
                "status": "blocked",
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
            recommendations.append(
                "Install the denoise provider in this runtime: "
                + _install_command(config, python_executable)
            )
            continue
        try:
            capabilities = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            environments[python_executable] = {
                "status": "blocked",
                "error": f"Provider returned invalid JSON: {error}",
            }
            continue
        compatibility_issues = validate_provider_capabilities(config, capabilities)
        if compatibility_issues:
            environments[python_executable] = {
                "status": "blocked",
                "error": "; ".join(compatibility_issues),
                "capabilities": capabilities,
            }
            recommendations.append(
                "Install a compatible denoise provider: "
                + _install_command(config, python_executable)
            )
            continue
        environments[python_executable] = {
            "status": "ready",
            "capabilities": capabilities,
        }

    methods: list[dict[str, Any]] = []
    for identifier, method in config.methods.items():
        issues: list[str] = []
        probe: dict[str, Any] | None = None
        python_record = environments.get(method.metadata.python_executable, {})
        if python_record.get("status") != "ready":
            issues.append("provider runtime unavailable")
        else:
            capability_methods = python_record["capabilities"].get("methods", [])
            capability = next(
                (
                    record
                    for record in capability_methods
                    if record.get("identifier") == identifier
                ),
                None,
            )
            if capability is None:
                issues.append("method missing from provider capabilities")
            elif not capability.get("available", False):
                issues.append("method dependency unavailable in provider runtime")
        if method.kind == "asn":
            checkpoint_value = method.metadata.checkpoint_path
            checkpoint = None if checkpoint_value is None else Path(checkpoint_value)
            if checkpoint is None or not checkpoint.is_file():
                issues.append("checkpoint missing")
            elif method.metadata.checkpoint_sha256:
                actual = _file_sha256(checkpoint)
                if actual != method.metadata.checkpoint_sha256:
                    issues.append("checkpoint SHA-256 mismatch")
        if not issues:
            with tempfile.TemporaryDirectory(
                prefix="denoise-harness-doctor-"
            ) as temporary:
                job_path = Path(temporary) / "probe.json"
                payload = {
                    "schema_version": WORKER_SCHEMA_VERSION,
                    "input_path": str(Path(temporary) / "unused.npy"),
                    "output_path": str(Path(temporary) / "unused.npz"),
                    "record_path": str(Path(temporary) / "unused.json"),
                    "comparison_policy": "clipped_0_1",
                    "options": {},
                    "method": {
                        "identifier": identifier,
                        "kind": method.kind,
                        "model_name": method.model_name,
                        "postprocess": method.postprocess,
                        "size_multiple": method.metadata.size_multiple,
                        "device_kind": method.metadata.device_kind,
                        "checkpoint_path": method.metadata.checkpoint_path,
                        "checkpoint_sha256": method.metadata.checkpoint_sha256,
                        "adapter_options": method.adapter_options,
                    },
                }
                job_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    provider_command(
                        config,
                        method.metadata.python_executable,
                        "--probe",
                        job_path,
                    ),
                    cwd=provider_working_directory(config),
                    env=provider_environment(config),
                    text=True,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                if completed.returncode != 0:
                    issues.append("provider probe failed")
                    probe = {
                        "available": False,
                        "error": completed.stderr.strip()
                        or completed.stdout.strip(),
                    }
                    recommendations.append(
                        f"Install or configure dependencies for method {identifier}."
                    )
                else:
                    probe = json.loads(completed.stdout)
        methods.append(
            {
                "identifier": identifier,
                "status": "blocked" if issues else "ready",
                "issues": issues,
                "probe": probe,
            }
        )
    ready_count = sum(record["status"] == "ready" for record in methods)
    if ready_count == len(methods):
        status = "ready"
    elif ready_count:
        status = "degraded"
    else:
        status = "blocked"
    return {
        "status": status,
        "provider": {
            "name": config.provider.name,
            "module": config.provider.module,
            "source_root": (
                None
                if config.provider.source_root is None
                else str(config.provider.source_root)
            ),
        },
        "environments": environments,
        "methods": methods,
        "recommendations": list(dict.fromkeys(recommendations)),
    }
