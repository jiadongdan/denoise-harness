"""Stable JSON CLI for the cross-agent scientific denoise harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .config import load_harness_config
from .image_io import inspect_input
from .initialization import initialize_config, parse_assignments
from .provider import default_provider_install_command, doctor
from .workflow import DenoiseHarness


DEFAULT_CONFIG_NAME = "denoise-harness.json"


def _config_path(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = os.environ.get("DENOISE_HARNESS_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / DEFAULT_CONFIG_NAME).resolve()


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Configuration path. Defaults to DENOISE_HARNESS_CONFIG or "
            f"./{DEFAULT_CONFIG_NAME}."
        ),
    )


def _load_method_options(arguments: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if arguments.options_json and arguments.options_file:
        raise ValueError("Use either --options-json or --options-file, not both.")
    if arguments.options_file:
        payload = json.loads(arguments.options_file.read_text(encoding="utf-8"))
    elif arguments.options_json:
        payload = json.loads(arguments.options_json)
    else:
        payload = {}
    if not isinstance(payload, dict) or not all(
        isinstance(value, dict) for value in payload.values()
    ):
        raise TypeError("Method options must map identifiers to JSON objects.")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, prog="denoise")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="Discover denoise-learn and create a config.")
    initialize.add_argument("--output", type=Path, default=Path(DEFAULT_CONFIG_NAME))
    initialize.add_argument("--provider-python", default=sys.executable)
    initialize.add_argument("--checkpoint", action="append", default=[], metavar="METHOD=PATH")
    initialize.add_argument(
        "--checkpoint-state-key", action="append", default=[], metavar="METHOD=KEY"
    )
    initialize.add_argument(
        "--checkpoint-schema-version",
        action="append",
        default=[],
        metavar="METHOD=VERSION",
    )
    initialize.add_argument(
        "--checkpoint-model-name", action="append", default=[], metavar="METHOD=NAME"
    )
    initialize.add_argument("--force", action="store_true")

    doctor_parser = commands.add_parser("doctor", help="Check configuration and runtime readiness.")
    _add_config_argument(doctor_parser)

    methods = commands.add_parser("methods", help="List configured denoising methods.")
    _add_config_argument(methods)
    methods.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True)

    inspect = commands.add_parser("inspect", help="Inspect one input image without denoising.")
    _add_config_argument(inspect)
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--normalization", default="minmax_0_1")

    run = commands.add_parser("run", help="Execute one reproducible denoising run.")
    _add_config_argument(run)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--reference", type=Path)
    run.add_argument("--methods", nargs="+", default=None)
    run.add_argument("--options-json")
    run.add_argument("--options-file", type=Path)
    run.add_argument(
        "--auto-ml-parameters", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--search-budget", type=int)
    run.add_argument("--comparison-policy")
    run.add_argument("--input-normalization")
    run.add_argument("--reference-normalization")
    run.add_argument("--output-root", type=Path)

    reproduce = commands.add_parser("reproduce", help="Re-run a prior run record.")
    _add_config_argument(reproduce)
    reproduce.add_argument("--record", type=Path, required=True)
    reproduce.add_argument("--output-root", type=Path)
    return parser


def _missing_config_report(path: Path) -> dict[str, Any]:
    return {
        "status": "blocked",
        "config_path": str(path),
        "issues": ["The denoise-harness configuration does not exist."],
        "recommendations": [
            f'denoise init --output "{path}"',
            "If denoise-learn is missing, install it from the public repository: "
            + default_provider_install_command("python"),
        ],
    }


def _execute(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "init":
        return initialize_config(
            arguments.output,
            provider_python=arguments.provider_python,
            checkpoints=parse_assignments(arguments.checkpoint, "checkpoint"),
            state_keys=parse_assignments(
                arguments.checkpoint_state_key, "checkpoint-state-key"
            ),
            schema_versions=parse_assignments(
                arguments.checkpoint_schema_version, "checkpoint-schema-version"
            ),
            checkpoint_model_names=parse_assignments(
                arguments.checkpoint_model_name, "checkpoint-model-name"
            ),
            force=bool(arguments.force),
        )

    config_path = _config_path(arguments.config)
    if arguments.command == "doctor" and not config_path.is_file():
        return _missing_config_report(config_path)
    config = load_harness_config(config_path)
    harness = DenoiseHarness(config)
    if arguments.command == "doctor":
        return doctor(config)
    if arguments.command == "methods":
        return {"status": "ok", "methods": harness.list_methods(probe=arguments.probe)}
    if arguments.command == "inspect":
        _, record = inspect_input(arguments.input, arguments.normalization)
        return {"status": "ok", "input": record}
    if arguments.command == "reproduce":
        return harness.reproduce(arguments.record.resolve(), output_root=arguments.output_root)

    methods = arguments.methods
    if methods == ["all"]:
        methods = None
    defaults = config.defaults
    plan = harness.build_plan(
        methods,
        _load_method_options(arguments),
        auto_ml_parameters=bool(arguments.auto_ml_parameters),
        search_budget=int(
            arguments.search_budget
            if arguments.search_budget is not None
            else defaults.get("search_budget", 6)
        ),
        comparison_policy=str(
            arguments.comparison_policy
            if arguments.comparison_policy is not None
            else defaults.get("comparison_policy", "clipped_0_1")
        ),
        input_normalization=str(
            arguments.input_normalization
            if arguments.input_normalization is not None
            else defaults.get("input_normalization", "minmax_0_1")
        ),
        reference_normalization=str(
            arguments.reference_normalization
            if arguments.reference_normalization is not None
            else defaults.get("reference_normalization", "minmax_0_1")
        ),
    )
    return harness.run(
        arguments.input,
        reference_path=arguments.reference,
        plan=plan,
        output_root=arguments.output_root,
    )


def main() -> None:
    arguments = _parser().parse_args()
    try:
        result = _execute(arguments)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
