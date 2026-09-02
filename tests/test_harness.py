"""Tests for the standalone cross-agent scientific denoise harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

from denoise_harness.config import load_harness_config
from denoise_harness.diagnostics import (
    heuristic_candidate_score,
    no_reference_diagnostics,
    reference_metrics,
)
from denoise_harness.image_io import normalize_image
from denoise_harness.initialization import initialize_config, parse_assignments
from denoise_harness.provider import (
    PROVIDER_REPOSITORY_URL,
    default_provider_install_command,
    doctor,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CJK_PATTERN = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3100-\u312f"
    r"\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uff00-\uffef\U00020000-\U0002fa1f]"
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE"}


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(REPOSITORY_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    return environment


def _capabilities() -> dict[str, object]:
    return {
        "contract_version": "denoise-learn-provider-v1",
        "provider": "denoise-learn",
        "provider_version": "0.1.0",
        "methods": [
            {
                "identifier": "mtflearn_fft",
                "kind": "mtflearn_fft",
                "family": "classical_fft",
                "available": True,
                "defaults": {"p": 0.01},
                "parameter_schema": {
                    "p": {"type": "number", "minimum": 0.001, "maximum": 1.0}
                },
            },
            {
                "identifier": "asn_gen1",
                "kind": "asn",
                "family": "deep_learning",
                "model_name": "asn_gen1",
                "device_kind": "cuda",
                "checkpoint_required": True,
            },
        ],
    }


def _missing_provider_config(directory: Path) -> Path:
    payload = json.loads(
        (REPOSITORY_ROOT / "configs" / "config.example.json").read_text(
            encoding="utf-8"
        )
    )
    payload["project_root"] = str(REPOSITORY_ROOT)
    payload["output_root"] = str(directory / "runs")
    payload["provider"]["module"] = "missing_denoise_provider.worker"
    path = directory / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repository_is_self_contained_but_does_not_vendor_denoise_learn() -> None:
    assert (REPOSITORY_ROOT / "SKILL.md").is_file()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert not (REPOSITORY_ROOT / "denoiselearn").exists()
    assert not list(REPOSITORY_ROOT.rglob("*.pt"))
    assert not list(REPOSITORY_ROOT.rglob("*.pth"))


def test_repository_content_is_english_only() -> None:
    violations: list[str] = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or (
            path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES
        ):
            continue
        if CJK_PATTERN.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert violations == []


def test_skill_is_host_and_workspace_neutral() -> None:
    content = (REPOSITORY_ROOT / "SKILL.md").read_text(encoding="utf-8")
    forbidden = ["Denoise benchmark", "denoise_agent", "C:\\", "D:\\"]
    assert all(value not in content for value in forbidden)


def test_example_config_is_portable() -> None:
    config = load_harness_config(REPOSITORY_ROOT / "configs" / "config.example.json")
    assert set(config.methods) == {"mtflearn_fft", "mtflearn_svd"}
    assert config.methods["mtflearn_fft"].defaults["p"] == 0.01
    assert PROVIDER_REPOSITORY_URL in str(config.provider.install_hint)
    assert config.source_path.is_absolute()


def test_init_discovers_classical_method_and_guides_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "denoise-harness.json"
        result = initialize_config(output, capabilities=_capabilities())
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert result["enabled_methods"] == ["mtflearn_fft"]
        assert result["checkpoint_guidance"][0]["identifier"] == "asn_gen1"
        assert payload["schema_version"] == "scientific-denoise-harness-config-v1"
        assert payload["methods"][0]["python_executable"] == str(Path(sys.executable).resolve())
        assert PROVIDER_REPOSITORY_URL in payload["provider"]["install_hint"]
        assert result["next_command"].startswith("denoise doctor")
        assert result["checkpoint_guidance"][0]["action"].startswith(
            "Run denoise init --force"
        )


def test_init_hashes_user_checkpoint_without_copying_it() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        checkpoint = directory / "model.bin"
        checkpoint.write_bytes(b"checkpoint")
        output = directory / "denoise-harness.json"
        initialize_config(
            output,
            checkpoints={"asn_gen1": str(checkpoint)},
            capabilities=_capabilities(),
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        method = next(item for item in payload["methods"] if item["identifier"] == "asn_gen1")
        assert method["checkpoint_path"] == str(checkpoint.resolve())
        assert len(method["checkpoint_sha256"]) == 64
        assert checkpoint.read_bytes() == b"checkpoint"
        assert not list(REPOSITORY_ROOT.rglob("model.bin"))


def test_doctor_guides_denoise_learn_installation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        config = load_harness_config(_missing_provider_config(Path(temporary)))
        report = doctor(config)
        assert report["status"] == "blocked"
        assert any("pip install" in item for item in report["recommendations"])
        assert any(PROVIDER_REPOSITORY_URL in item for item in report["recommendations"])


def test_default_provider_install_command_uses_public_github_repository() -> None:
    command = default_provider_install_command("python")
    assert PROVIDER_REPOSITORY_URL in command
    assert "denoise-learn[inference]" in command


def test_normalization_and_metric_boundaries() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    unit, record = normalize_image(image, "minmax_0_1")
    assert record["original_range"] == [0.0, 63.0]
    assert float(unit.min()) == 0.0
    assert float(unit.max()) == 1.0
    metrics = reference_metrics(unit, np.clip(unit + 0.01, 0.0, 1.0))
    diagnostics = no_reference_diagnostics(unit, unit)
    assert "psnr" in metrics
    assert "psnr" not in diagnostics
    assert np.isfinite(heuristic_candidate_score(diagnostics))


def test_assignment_parser_rejects_invalid_values() -> None:
    assert parse_assignments(["asn_gen1=model.pt"], "checkpoint") == {
        "asn_gen1": "model.pt"
    }
    try:
        parse_assignments(["asn_gen1"], "checkpoint")
    except ValueError as error:
        assert "METHOD=VALUE" in str(error)
    else:
        raise AssertionError("Invalid assignment was accepted.")


def test_cli_exposes_only_the_six_public_workflow_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "denoise_harness.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
        env=_source_environment(),
    )
    assert completed.returncode == 0
    for command in ("init", "doctor", "methods", "inspect", "run", "reproduce"):
        assert command in completed.stdout
    assert "validate-config" not in completed.stdout


def test_doctor_without_config_returns_actionable_json() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "missing.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "denoise_harness.cli",
                "doctor",
                "--config",
                str(missing),
            ],
            text=True,
            capture_output=True,
            check=False,
            env=_source_environment(),
        )
        assert completed.returncode == 0
        payload = json.loads(completed.stdout)
        assert payload["status"] == "blocked"
        assert any("denoise init" in item for item in payload["recommendations"])
        assert any(PROVIDER_REPOSITORY_URL in item for item in payload["recommendations"])
