---
name: denoise-harness
description: Run, compare, evaluate, and reproduce installed denoising methods on single-channel scientific or microscopy images. Use when the user asks to denoise an image, compare available methods, tune bounded FFT/SVD parameters, inspect residual risks, or reproduce a prior run. Requires a local denoise-harness CLI and guides users when the external denoise-learn provider or checkpoints are not installed. Do not use for model training, checkpoint downloads, or generic photo editing.
metadata:
  version: "0.1.0"
  product: "denoise-harness"
---

# Denoise Harness

Use the local `denoise` CLI as the deterministic execution surface. The host agent
resolves user intent, invokes commands, and explains their JSON output. Pixel
computation remains in the separately installed `denoise-learn` provider.

## Cold Start

1. Run `denoise doctor`. If the command is missing, explain that the Harness Python
   package must be installed from this repository; do not install without permission.
2. If `doctor` or `init` reports that `denoise-learn` is missing, present its exact
   public GitHub installation recommendation:
   `python -m pip install "denoise-learn[inference] @ git+https://github.com/jiadongdan/denoise-learn.git@main"`.
   Never copy source from another checkout or install a dependency without user
   authorization.
3. If no config exists, run `denoise init`. Classical methods that need no checkpoint
   may be configured immediately. Report checkpoint guidance for optional neural
   methods; never search for or download checkpoint files.
4. Run `denoise doctor` again and stop when its status is `blocked`.

## Workflow

1. Run `denoise methods` and use only reported available identifiers, parameter
   schemas, devices, and checkpoints.
2. Run `denoise inspect --input <path>` before execution. Reject unsupported,
   multichannel, nonnumeric, or nonfinite inputs.
3. Resolve the requested methods, optional clean reference, explicit options, and
   optional bounded FFT/SVD search.
4. Run `denoise run`. Never bypass validation, reinterpret checkpoint identity, or
   silently replace a CUDA-only method with a CPU path.
5. Preserve partial successes when one method fails. Report the run directory,
   selected parameters, completed and failed methods, valid metrics, uncertainty,
   and scientific-risk warnings.
6. Use `denoise reproduce --record <run_record.json>` for reproduction. Never
   overwrite an earlier run.

## Scientific Invariants

- Preserve raw, comparison, and residual outputs separately.
- Report PSNR, SSIM, MSE, or MAE only with an explicit clean reference and recorded
  `data_range=1.0` comparison policy.
- Without a clean reference, label selection as heuristic and allow uncertainty.
- Treat atomic columns, defects, interfaces, or periodic signal in residuals as risk
  evidence even when a global metric improves.
- Keep unpublished images local unless the user explicitly permits an upload.
- Treat CLI discovery, validation, versioned JSON, and run records as the source of
  truth. Do not duplicate their schemas in the Skill.

## Compatibility

This Skill uses no host-specific tool names, multi-agent primitives, repository paths,
or fixed Python environments. A host that can read local files, execute shell commands,
and parse JSON can run it sequentially on Windows, macOS, or Linux. Report a missing
capability instead of inventing a fallback.
