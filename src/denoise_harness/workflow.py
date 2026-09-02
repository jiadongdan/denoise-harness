"""Cross-environment workflow, parameter search, artifacts, and run records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import subprocess
import tempfile
import uuid
from typing import Any

import numpy as np

from .config import HarnessConfig, MethodConfig
from .contracts import CONTRACT_VERSION, DenoiseResult, RunPlan
from .diagnostics import (
    heuristic_candidate_score,
    no_reference_diagnostics,
    reference_metrics,
)
from .image_io import file_sha256, inspect_input, save_preview
from .search import generate_candidates
from .provider import (
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
    provider_command,
    provider_environment,
    provider_working_directory,
)


RUN_RECORD_SCHEMA_VERSION = "scientific-denoise-run-record-v1"
COMPARISON_POLICIES = {"raw_unit_space", "clipped_0_1", "minmax_0_1"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_scalar(value: Any, specification: dict[str, Any], label: str) -> Any:
    value_type = specification.get("type")
    if value_type == "integer":
        converted: Any = int(value)
    elif value_type == "number":
        converted = float(value)
    else:
        converted = value
    if "minimum" in specification and converted < specification["minimum"]:
        raise ValueError(f"{label} is below its minimum.")
    if "maximum" in specification and converted > specification["maximum"]:
        raise ValueError(f"{label} is above its maximum.")
    return converted


def validate_method_options(method: MethodConfig, options: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize options against one manifest parameter schema."""
    merged = dict(method.defaults)
    merged.update(options)
    schema = method.metadata.parameter_schema
    allowed = {key for key in schema if key != "candidate_pairs"}
    unknown = sorted(set(merged).difference(allowed))
    if unknown:
        raise ValueError(
            f"Unknown options for {method.metadata.identifier}: {', '.join(unknown)}"
        )
    normalized: dict[str, Any] = {}
    for name, value in merged.items():
        specification = schema.get(name, {})
        if not isinstance(specification, dict):
            raise TypeError(f"Parameter schema for {name} must be an object.")
        normalized[name] = _validate_scalar(
            value, specification, f"{method.metadata.identifier}.{name}"
        )
    return normalized


class DenoiseHarness:
    """Run registered denoisers through isolated, environment-specific workers."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def list_methods(self, *, probe: bool = False) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for method in self.config.methods.values():
            record = {
                **method.metadata.to_dict(),
                "kind": method.kind,
                "model_name": method.model_name,
                "postprocess": method.postprocess,
                "defaults": method.defaults,
                "available": True,
            }
            if probe:
                try:
                    record["probe"] = self._probe_method(method)
                except Exception as error:
                    record["available"] = False
                    record["probe"] = {
                        "available": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            records.append(record)
        return records

    def _probe_method(self, method: MethodConfig) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="denoise-harness-probe-") as temporary:
            probe_dir = Path(temporary)
            job_path = probe_dir / f"{method.metadata.identifier}.json"
            payload = self._method_job_payload(
                method,
                {},
                probe_dir / "unused.npy",
                probe_dir / "unused.npz",
                probe_dir / "unused-record.json",
                "clipped_0_1",
            )
            _write_json(job_path, payload)
            completed = subprocess.run(
                provider_command(
                    self.config,
                    method.metadata.python_executable,
                    "--probe",
                    job_path,
                ),
                cwd=provider_working_directory(self.config),
                env=provider_environment(self.config),
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "Method probe failed.")
            return json.loads(completed.stdout)

    def build_plan(
        self,
        methods: list[str] | None,
        method_options: dict[str, dict[str, Any]] | None,
        *,
        auto_ml_parameters: bool,
        search_budget: int,
        comparison_policy: str,
        input_normalization: str,
        reference_normalization: str,
    ) -> RunPlan:
        selected = list(self.config.methods) if methods is None else list(dict.fromkeys(methods))
        if not selected:
            raise ValueError("Select at least one denoising method.")
        unknown = sorted(set(selected).difference(self.config.methods))
        if unknown:
            raise ValueError(f"Unknown denoising methods: {', '.join(unknown)}")
        if comparison_policy not in COMPARISON_POLICIES:
            raise ValueError(f"Unsupported comparison policy: {comparison_policy}")
        if search_budget < 1:
            raise ValueError("Search budget must be positive.")
        supplied = method_options or {}
        extra = sorted(set(supplied).difference(selected))
        if extra:
            raise ValueError(f"Options supplied for unselected methods: {', '.join(extra)}")
        normalized: dict[str, dict[str, Any]] = {}
        for identifier in selected:
            method = self.config.methods[identifier]
            requested = dict(supplied.get(identifier, {}))
            if auto_ml_parameters and method.kind in {"mtflearn_fft", "mtflearn_svd"} and not requested:
                normalized[identifier] = {}
            else:
                normalized[identifier] = validate_method_options(method, requested)
        return RunPlan(
            methods=tuple(selected),
            method_options=normalized,
            auto_ml_parameters=bool(auto_ml_parameters),
            search_budget=int(search_budget),
            comparison_policy=comparison_policy,
            input_normalization=input_normalization,
            reference_normalization=reference_normalization,
        )
    def _method_job_payload(
        self,
        method: MethodConfig,
        options: dict[str, Any],
        input_path: Path,
        output_path: Path,
        record_path: Path,
        comparison_policy: str,
    ) -> dict[str, Any]:
        metadata = method.metadata
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "record_path": str(record_path),
            "comparison_policy": comparison_policy,
            "options": options,
            "method": {
                "identifier": metadata.identifier,
                "kind": method.kind,
                "model_name": method.model_name,
                "postprocess": method.postprocess,
                "size_multiple": metadata.size_multiple,
                "device_kind": metadata.device_kind,
                "checkpoint_path": metadata.checkpoint_path,
                "checkpoint_sha256": metadata.checkpoint_sha256,
                "adapter_options": method.adapter_options,
            },
        }

    def _call_worker(
        self,
        method: MethodConfig,
        options: dict[str, Any],
        input_path: Path,
        directory: Path,
        comparison_policy: str,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        job_path = directory / "job.json"
        output_path = directory / "worker_output.npz"
        record_path = directory / "worker_record.json"
        stdout_path = directory / "worker_stdout.txt"
        stderr_path = directory / "worker_stderr.txt"
        _write_json(
            job_path,
            self._method_job_payload(
                method,
                options,
                input_path,
                output_path,
                record_path,
                comparison_policy,
            ),
        )
        completed = subprocess.run(
            provider_command(
                self.config,
                method.metadata.python_executable,
                "--job",
                job_path,
            ),
            cwd=provider_working_directory(self.config),
            env=provider_environment(self.config),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Worker for {method.metadata.identifier} failed with exit code "
                f"{completed.returncode}. See {stderr_path}."
            )
        if not output_path.is_file() or not record_path.is_file():
            raise RuntimeError("Worker completed without required artifacts.")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
            raise RuntimeError("Provider result contract version mismatch.")
        if record.get("provider") != self.config.provider.name:
            raise RuntimeError("Provider result identity mismatch.")
        return record, output_path, record_path

    def _warnings_from_diagnostics(self, diagnostics: dict[str, float]) -> list[str]:
        warnings: list[str] = []
        if diagnostics["gradient_rms_ratio"] < 0.55:
            warnings.append("Output gradient energy is strongly reduced; inspect possible structure loss.")
        if abs(diagnostics["residual_input_correlation"]) > 0.35:
            warnings.append("Residual is strongly correlated with the input; inspect removed structure.")
        if diagnostics["residual_periodic_peak_ratio"] > 80.0:
            warnings.append("Residual contains concentrated periodic frequency peaks.")
        return warnings

    def _materialize_result(
        self,
        method: MethodConfig,
        options: dict[str, Any],
        input_image: np.ndarray,
        reference: np.ndarray | None,
        worker_record: dict[str, Any],
        worker_output: Path,
        worker_record_path: Path,
        method_dir: Path,
        search_record: dict[str, Any] | None,
    ) -> DenoiseResult:
        method_dir.mkdir(parents=True, exist_ok=True)
        with np.load(worker_output, allow_pickle=False) as arrays:
            raw = np.asarray(arrays["raw"], dtype=np.float32)
            comparison = np.asarray(arrays["comparison"], dtype=np.float32)
        residual = input_image - comparison
        raw_path = method_dir / "raw_output.npy"
        comparison_path = method_dir / "comparison_output.npy"
        residual_path = method_dir / "residual.npy"
        np.save(raw_path, raw)
        np.save(comparison_path, comparison)
        np.save(residual_path, residual.astype(np.float32))
        save_preview(method_dir / "comparison_preview.png", comparison)
        signed_limit = float(np.max(np.abs(residual)))
        residual_preview = (
            np.full(residual.shape, 0.5, dtype=np.float32)
            if signed_limit == 0.0
            else np.clip(0.5 + residual / (2.0 * signed_limit), 0.0, 1.0)
        )
        save_preview(method_dir / "residual_preview.png", residual_preview)
        diagnostics = no_reference_diagnostics(input_image, comparison)
        metrics = {} if reference is None else reference_metrics(reference, comparison)
        warnings = self._warnings_from_diagnostics(diagnostics)
        if reference is None:
            warnings.append(
                "No clean reference was supplied; PSNR, SSIM, MSE, and MAE were not computed."
            )
        result = DenoiseResult(
            identifier=method.metadata.identifier,
            metadata=method.metadata,
            options=options,
            raw_output_path=str(raw_path.resolve()),
            comparison_output_path=str(comparison_path.resolve()),
            residual_path=str(residual_path.resolve()),
            runtime_seconds=float(worker_record["runtime_seconds"]),
            raw_output_range=tuple(float(value) for value in worker_record["raw_output_range"]),
            comparison_output_range=tuple(
                float(value) for value in worker_record["comparison_output_range"]
            ),
            transforms=list(worker_record.get("transforms", [])),
            diagnostics=diagnostics,
            metrics=metrics,
            warnings=warnings,
            worker_record_path=str(worker_record_path.resolve()),
            search=search_record,
        )
        _write_json(method_dir / "result.json", result.to_dict())
        return result

    def _search_method(
        self,
        method: MethodConfig,
        input_image: np.ndarray,
        input_path: Path,
        reference: np.ndarray | None,
        method_dir: Path,
        plan: RunPlan,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
        candidates = generate_candidates(method, input_image, plan.search_budget)
        records: list[dict[str, Any]] = []
        best: tuple[float, int, dict[str, Any], dict[str, Any], Path, Path] | None = None
        for index, candidate in enumerate(candidates):
            options = validate_method_options(method, candidate)
            candidate_dir = method_dir / "search" / f"candidate_{index:03d}"
            worker_record, worker_output, worker_record_path = self._call_worker(
                method,
                options,
                input_path,
                candidate_dir,
                plan.comparison_policy,
                timeout_seconds,
            )
            with np.load(worker_output, allow_pickle=False) as arrays:
                comparison = np.asarray(arrays["comparison"], dtype=np.float32)
            diagnostics = no_reference_diagnostics(input_image, comparison)
            metrics = {} if reference is None else reference_metrics(reference, comparison)
            score = (
                float(metrics["psnr"])
                if reference is not None
                else heuristic_candidate_score(diagnostics)
            )
            candidate_record = {
                "index": index,
                "options": options,
                "score": score,
                "score_policy": "reference_psnr" if reference is not None else "no_reference_heuristic_v1",
                "metrics": metrics,
                "diagnostics": diagnostics,
                "worker_record_path": str(worker_record_path.resolve()),
                "worker_output_path": str(worker_output.resolve()),
            }
            records.append(candidate_record)
            if best is None or (score, -index) > (best[0], -best[1]):
                best = (score, index, options, worker_record, worker_output, worker_record_path)
        assert best is not None
        search_record = {
            "automatic": True,
            "budget": plan.search_budget,
            "executed_candidates": len(records),
            "score_policy": records[0]["score_policy"],
            "selected_index": best[1],
            "selected_options": best[2],
            "selected_score": best[0],
            "candidates": records,
            "warning": (
                None
                if reference is not None
                else "No-reference selection is heuristic and does not establish ground-truth quality."
            ),
        }
        _write_json(method_dir / "search.json", search_record)
        return best[2], search_record, best[4], best[5]

    def run(
        self,
        input_path: str | Path,
        *,
        reference_path: str | Path | None = None,
        plan: RunPlan,
        output_root: str | Path | None = None,
    ) -> dict[str, Any]:
        input_image, input_record = inspect_input(input_path, plan.input_normalization)
        reference = None
        reference_record = None
        if reference_path is not None:
            reference, reference_record = inspect_input(
                reference_path, plan.reference_normalization
            )
            if reference.shape != input_image.shape:
                raise ValueError("Clean reference shape does not match the input shape.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        root = self.config.output_root if output_root is None else Path(output_root).resolve()
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        normalized_input_path = run_dir / "input_unit.npy"
        np.save(normalized_input_path, input_image)
        save_preview(run_dir / "input_preview.png", input_image)
        if reference is not None:
            np.save(run_dir / "reference_unit.npy", reference)
            save_preview(run_dir / "reference_preview.png", reference)
        _write_json(run_dir / "plan.json", plan.to_dict())
        timeout_seconds = int(self.config.defaults.get("timeout_seconds", 600))
        results: list[DenoiseResult] = []
        failures: list[dict[str, Any]] = []
        for identifier in plan.methods:
            method = self.config.methods[identifier]
            method_dir = run_dir / "methods" / identifier
            try:
                requested_options = dict(plan.method_options.get(identifier, {}))
                use_search = (
                    plan.auto_ml_parameters
                    and method.kind in {"mtflearn_fft", "mtflearn_svd"}
                    and not requested_options
                )
                if use_search:
                    options, search_record, worker_output, worker_record_path = self._search_method(
                        method,
                        input_image,
                        normalized_input_path,
                        reference,
                        method_dir,
                        plan,
                        timeout_seconds,
                    )
                    worker_record = json.loads(worker_record_path.read_text(encoding="utf-8"))
                else:
                    options = validate_method_options(method, requested_options)
                    search_record = None
                    worker_record, worker_output, worker_record_path = self._call_worker(
                        method,
                        options,
                        normalized_input_path,
                        method_dir / "execution",
                        plan.comparison_policy,
                        timeout_seconds,
                    )
                results.append(
                    self._materialize_result(
                        method,
                        options,
                        input_image,
                        reference,
                        worker_record,
                        worker_output,
                        worker_record_path,
                        method_dir,
                        search_record,
                    )
                )
            except Exception as error:
                failure = {
                    "identifier": identifier,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(method_dir / "failure.json", failure)
        status = "completed" if results and not failures else "partial" if results else "failed"
        record = {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "config_path": str(self.config.source_path),
            "config_sha256": file_sha256(self.config.source_path),
            "input": input_record,
            "reference": reference_record,
            "plan": plan.to_dict(),
            "results": [result.to_dict() for result in results],
            "failures": failures,
            "run_directory": str(run_dir.resolve()),
        }
        _write_json(run_dir / "run_record.json", record)
        report_lines = [
            "# Scientific image denoise run",
            "",
            f"- Run ID: `{run_id}`",
            f"- Status: `{status}`",
            f"- Methods completed: {len(results)}",
            f"- Methods failed: {len(failures)}",
            "",
        ]
        for result in results:
            report_lines.extend(
                [
                    f"## {result.identifier}",
                    "",
                    f"- Options: `{json.dumps(result.options, sort_keys=True)}`",
                    f"- Runtime: `{result.runtime_seconds:.6f}` seconds",
                    f"- Metrics: `{json.dumps(result.metrics, sort_keys=True)}`",
                    f"- Warnings: `{json.dumps(result.warnings)}`",
                    "",
                ]
            )
        (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
        return record

    def reproduce(self, record_path: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
        """Re-run the exact selected method options from a prior record."""
        prior = json.loads(Path(record_path).read_text(encoding="utf-8"))
        if prior.get("schema_version") != RUN_RECORD_SCHEMA_VERSION:
            raise ValueError("Unsupported run record schema for reproduction.")
        options = {
            str(result["identifier"]): dict(result["options"])
            for result in prior.get("results", [])
        }
        methods = list(options)
        if not methods:
            raise ValueError("Prior run record contains no successful results to reproduce.")
        prior_plan = prior["plan"]
        plan = self.build_plan(
            methods,
            options,
            auto_ml_parameters=False,
            search_budget=int(prior_plan["search_budget"]),
            comparison_policy=str(prior_plan["comparison_policy"]),
            input_normalization=str(prior_plan["input_normalization"]),
            reference_normalization=str(prior_plan["reference_normalization"]),
        )
        reference = None if prior.get("reference") is None else prior["reference"]["path"]
        return self.run(
            prior["input"]["path"],
            reference_path=reference,
            plan=plan,
            output_root=output_root,
        )
