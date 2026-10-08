# Validation evidence

These are actual local runs on 2026-10-08, not projected throughput. The checkpoint is Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, already present locally. The final code defaults to packed QKV/gate-up and FP32 on CPU/MPS.

## Relevant successful gates

| Report | Scope | Result |
|---|---|---|
| [CPU fused64](qwen25-05b-cpu-fp32-fused64.json) | FP32, packed projections, prompt lookup enabled, 2 prompts ×64 tokens, cold plus warm/batched | Logits close; generated token IDs equal to HF cached-forward greedy reference |
| [MPS fused64](qwen25-05b-mps-fp32-fused64.json) | Same checks on Apple GPU FP32 | Logits close; generated token IDs equal |
| [Real HTTP](real-http-smoke.json) | Loaded real model, readiness, JSON and SSE, usage and repeated prefix | Nonempty equal answers, equal usage, 32 reused tokens, no active requests left |

The default SDPA path was exercised. These runs do not validate Triton/CUDA, large-model capacity, other architectures, stability under sustained load or a speedup over competing engines. Elapsed time in correctness reports includes reference inference and setup, so it is **not** a Flow inference benchmark.

[Local release checks](release-check.json) record 27 discovered tests: 26 passed and the CUDA-only test skipped, along with parsed Python sources, local documentation links and inspected build archives. The source archive was subsequently rebuilt to include that check report; archive sizes in the report describe the checked pre-report build, not an immutable checksum of the final archive.

## H100 preparation milestone

[Preparation checks](h100-preparation-checks-2026-10-08.json) record 52 tests, 51 passes and one CUDA skip. [Real CPU regression](qwen25-05b-cpu-fp32-h100-preparation.json) repeats the fixed64/prompt-lookup gate after releasing reference KV tensors before loading Flow. This avoids carrying reference-model memory into Flow's CUDA memory measurements. The standalone CUDA gate was also invoked on this Mac and correctly refused to pass without CUDA.

The live public-HF metadata staging attempt timed out on the local direct network; no weights were downloaded and no live staging success is claimed. Test the CPU/cloud networking before starting billed GPU work. Mocked metadata tests cover pinning and failure handling, not network availability. The preparation archive report describes the checked build; later documentation/report additions change archive contents and sizes.

## Failed experiments retained

- [Initial CPU run](qwen25-05b-cpu-fp32.json): logits matched, output comparison failed because HF `generate` inherited the checkpoint's repetition penalty while Flow used plain greedy decoding. The validation harness now uses direct cached forward and argmax.
- [Initial 48-token draft run](qwen25-05b-cpu-fp32-draft.json): matching overlapping token prefixes but differing output lengths due to HF fallback EOS stopping. Fixed-length forward comparison removed this benchmark-policy mismatch. [CPU fixed64](qwen25-05b-cpu-fp32-fixed64.json) passed afterward.
- [MPS FP16 fixed64](qwen25-05b-mps-fp16-fixed64.json): logits passed the reduced-precision tolerance but strict generated-token equality failed. This is an unqualified configuration for exact-output claims; it is not hidden behind a “supported GPU” label. Auto dtype on MPS remains FP32. This report predates projection packing and does not validate the final FP16 path.

Earlier short successful reports are retained as intermediate evidence; the final successful references are the fused64 reports above. Tests do not prove correctness for arbitrary prompts. Increase workload diversity before widening the support claim.

## Runtime observations

PyTorch 2.12.1, Transformers 4.51.3, Python 3.10.20 in the existing local environment. MPS was unavailable inside the restricted execution sandbox but available on the actual Mac; the Apple GPU checks ran with host device access. No packages, weights or paid compute were downloaded for these validations. The temporary HTTP test server was shut down afterward.

The CUDA unit test is explicitly skipped when no CUDA GPU is available. A passing CPU suite with that skip must never be described as a passed CUDA kernel test.

Transformers 4.51.3 prints a sliding-window warning when Qwen's `sliding_window` field is populated even if `use_sliding_window=false`. The validated checkpoint disables that flag; its reference attention path remains full attention. Flow rejects checkpoints with an active sliding-window setting.
