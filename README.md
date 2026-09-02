# denoise-harness

`denoise-harness` is a cross-agent Skill and deterministic CLI for reproducible
single-channel scientific image denoising. It lets a general-purpose host agent such
as Codex, Claude Code, WorkBuddy, Cursor, or another shell-capable agent inspect an
image, discover installed methods, execute comparable denoisers, preserve artifacts,
and explain scientific risks.

This repository is the new cross-platform product. The local `denoise-agent`
directory in the development workspace is the legacy prototype; this repository does
not import it or require it at runtime.

## Architecture

```text
Host general agent / LLM runtime
            |
            v
denoise-harness Skill + `denoise` CLI
            |
            v
separately installed denoise-learn provider
            |
            v
user-managed optional checkpoints
```

The repository intentionally contains neither `denoise-learn` source code nor model
checkpoints. Missing dependencies are reported with explicit installation or
configuration guidance; the Harness does not silently install software, download
weights, or upload images.

## Requirements

- Python 3.10 or newer
- Local file and shell access
- `denoise-learn[inference]` for numerical execution
- CUDA only for methods whose discovered capability requires it
- User-supplied checkpoints for neural methods that require weights

## Install

From a cloned checkout:

```bash
git clone https://github.com/jiadongdan/denoise-harness.git
cd denoise-harness
python -m pip install .
```

Install the external numerical provider from its public GitHub repository:

```bash
python -m pip install "denoise-learn[inference] @ git+https://github.com/jiadongdan/denoise-learn.git@main"
```

Provider source: [jiadongdan/denoise-learn](https://github.com/jiadongdan/denoise-learn)

Then initialize and check the runtime:

```bash
denoise init
denoise doctor
```

If `denoise-learn` is not installed, `denoise init` returns structured JSON containing
the same public repository installation command. The Harness never installs it
without user authorization. Neural checkpoint paths can be added during
initialization without copying weights into this repository:

```bash
denoise init --force --checkpoint asn_gen1=/path/to/asn-gen1.pt --checkpoint asn_denoise=/path/to/asn-denoise.pt
```

## Install as an Agent Skill

The repository root is an Agent Skills-compatible package because it contains
`SKILL.md`. A cross-agent skills installer can place it in a supported host:

```bash
npx skills add jiadongdan/denoise-harness
```

Installing the Skill teaches the host the workflow; installing the Python package
provides the `denoise` command. These are separate, explicit steps.

## CLI

The public interface is intentionally small:

```text
denoise init        discover denoise-learn and create configuration
denoise doctor      validate provider, runtime, CUDA, and checkpoints
denoise methods     discover configured methods and parameter contracts
denoise inspect     inspect an input without running a denoiser
denoise run         execute methods and persist a reproducible run
denoise reproduce   repeat successful options from a prior run record
```

All commands emit machine-readable JSON and use nonzero exit codes for errors.
Configuration defaults to `DENOISE_HARNESS_CONFIG` or `./denoise-harness.json`.

Examples:

```bash
denoise methods
denoise inspect --input image.tif
denoise run --input image.tif --methods all
denoise run --input noisy.tif --reference clean.tif --methods mtflearn_fft mtflearn_svd
denoise reproduce --record denoise-runs/<run-id>/run_record.json
```

## Scientific safeguards

- Raw, comparison, and residual arrays are separate artifacts.
- Reference metrics are emitted only when an explicit clean reference is supplied.
- No-reference parameter selection is labeled heuristic, never ground-truth quality.
- Checkpoint identity, transforms, normalization, versions, and parameters are kept
  in the run record.
- Periodic or structured residual content is reported as possible structure loss.
- Unpublished images stay local unless the user explicitly permits an upload.

## Development

```bash
python -m pip install ".[dev]"
python -m pytest -q
```

CI covers Python 3.10 and 3.12 on Linux, Windows, and macOS. Unit tests do not require
`denoise-learn`; provider integration is discovered and checked at runtime.

All repository content, including documentation, source code, comments, messages,
configuration, and tests, must be written in English.

## License

This project is released under the [MIT License](LICENSE).
