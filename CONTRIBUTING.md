# Contributing

Start with the [state walkthrough](docs/ARCHITECTURE.md). Correctness, measured benefit and clear boundaries are release requirements; a plausible design is not benchmark evidence.

## Development

```bash
conda env create -f environment.yml
conda activate flow-inference
python -B -m unittest discover -s tests -v
```

Changes to the forward pass must compare logits and greedy token IDs against an independent reference. Cache/scheduler changes need tests for partial blocks, reference ownership, cancellation, resource exhaustion and output equality after evictions. Speculative changes must cover acceptance, rejection, EOS and rejected-KV isolation. New model adapters must reject unsupported sub-configurations explicitly.

GPU kernels require actual hardware tests and a real checkpoint gate. A skipped CUDA test is not a pass. Performance submissions include immutable model/tokenizer revisions, exact execution dtype/hardware, launch commands, raw results and quality controls. Preserve failed experiments that explain the chosen defaults.

## Review and scope

Keep independent engine changes separate from visualization-platform changes. Do not vendor model weights, tokens, private prompts, datasets or copied upstream code without a license review. Public examples must use non-sensitive inputs. Check third-party and model licenses separately from this repository's MIT license.

Small reference models in tests are numerical/state fixtures, not the engine's workload or claimed performance target. Real inference always loads a real compatible checkpoint supplied by the user.

No telemetry, implicit model downloads or automatic GPU rental should be introduced without explicit user-facing controls. Add dependency versions and support limitations to the docs whenever interfaces change.
