# Fair inference comparison

The goal is lower latency at the **same output policy and model quality**, not merely a faster HTTP response. No comparison against vLLM or SGLang has been measured in the current release.

## Correctness gate

Run `tools/validate_model.py` first on the exact model/device/dtype/backend. It compares the engine with Transformers cached forward and fixed-length argmax, without checkpoint-specific sampling processors. It also checks warm prefixes and concurrent decode. Exact token equality is stricter than logit tolerance: near ties can fail even when numerical error is small. Such a run does not qualify for an exact-output performance claim.

CUDA requires `test_triton_paged_matches_reference` to execute rather than skip, followed by real model validation. Apple MPS success says nothing about the CUDA kernel. Tests with reduced random matrices check state invariants but do not replace a real-weight gate.

## Required experiment axes

Use the same immutable model/tokenizer revisions, dtype, chat template, generation rules, hardware, driver and workload. Log the full launch command without secrets. Fix output caps and compare EOS behavior. All engines must process the same prompts. Do not compare FP32 Flow to BF16 vLLM and attribute the difference to scheduling.

Run these independently:

1. Cold/disabled prefix cache, batch one, prompt lengths 128/2048/8192 where supported, fixed output cap.
2. Warm repeated prefix with a suffix change; distinguish reused token count from cache allocation.
3. Two/four simultaneous users and a long prompt arriving during another user's decode.
4. Copy/code-edit cases for prompt lookup, and non-repetitive cases that often reject drafts.
5. Flow ablations: packed/unpacked linears; prefix off/on; draft 0/2/4/8; SDPA/Triton on qualified CUDA.
6. Upstream normal optimized defaults AND matched-feature mechanism baselines. Do not disable useful upstream mechanisms just to create a victory.

The bundled HTTP script runs **sequential requests only**. Concurrent-load experiments require a separate load driver and are not implied by its results. Four bundled cases are smoke inputs, not a representative benchmark corpus. Use at least 30 measured repetitions per case after controlled warmup and report confidence intervals and p50/p95 for serious claims; the bundled comparison does not compute confidence intervals.

## Measure a running service

Copy `examples/manifest.template.json` to your experiment directory and fill every field. A model name alone is not a digest. `cache_regime` describes what you actually enforce; writing "cold" does not reset a server cache. For a no-prefix experiment start Flow with `--no-prefix-cache`; use the corresponding verified option in the pinned upstream version. Keep model-load time separate from request time.

```bash
python tools/benchmark_http.py run \
  --base-url http://127.0.0.1:8010/v1 --model flow-model \
  --manifest runs/flow-manifest.json --repeats 3 --max-tokens 64 \
  --output runs/flow-http.json
python tools/benchmark_http.py run \
  --base-url http://127.0.0.1:8000/v1 --model flow-model \
  --manifest runs/upstream-manifest.json --repeats 3 --max-tokens 64 \
  --output runs/upstream-http.json
python tools/benchmark_http.py compare runs/upstream-http.json runs/flow-http.json
```

Use a fresh filename; output reports are not overwritten. The client bypasses environment proxies. It does not download weights or call a cloud provider. Replace endpoints only with servers you intend to benchmark.

## Metric definitions

| Field | Meaning |
|---|---|
| `first_text_s` | Client time to first nonempty decoded text delta; not first token |
| `elapsed_s` | Client time until stream completion |
| `usage.completion_tokens` | Server-authoritative generated token count, not SSE chunks |
| `text_events_not_tokens` | Text delivery events; never presented as token throughput |
| `completion_tokens_per_total_second` | Output tokens / total request time, including prefill |
| engine `first_token.ttft_ms` | Server time from queue admission submission to first committed token |
| `step.elapsed_ms` | Measured forward/sampling step, not a kernel profiler trace |

Word/Unicode buffering affects first-text latency. The engine's greedy path transfers only the selected ID; stochastic sampling currently transfers logits to CPU. Different sampling modes have different costs. HTTP `/metrics` does not expose actual GPU utilization or energy.

`compare` requires matching experiment invariants, case/repetition pairs, complete responses, text, usage and finish reasons. If a run fails or returns different outputs, it refuses an aggregate speedup. The eligible result is a paired median total-latency ratio, not pure decode TPS, general SOTA evidence or a significance test.

## Before publishing a speedup

Archive raw results, actual launch commands, warmup policy, environment and model revision. Profile suspected bottlenecks with the appropriate hardware tools. Show negative cases alongside positive ones. Check memory peaks, long-run cancellation, overloaded queues and correctness after evictions. Run competitors in separate clean environments, on the same otherwise idle machine. Never treat successful CPU unit tests as evidence that an unexecuted GPU path is fast.
