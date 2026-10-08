# Evidence driven engine roadmap

The first release is an independent execution substrate, not a claim to replace mature serving systems. The target is fast individual responses AND efficient concurrent serving. Low load is one experiment axis, not a product boundary. Paging, caching and speculative decoding are established techniques; originality must come from a demonstrated new policy/kernel/design with controlled evidence.

## Performance contract across load levels

Evaluate closed-loop client concurrency 1/2/4/8/16/32 separately, including mixed prompt lengths. Report client first-text and total latency p50/p95, completed requests/s, authoritative output tokens/s, failures and device-memory peaks. Compare equal model, precision, hardware and output policy. Saturation tests must additionally use an open-loop arrival-rate driver; the bundled closed-loop tool cannot establish an arrival-rate SLO.

An optimization must disclose its whole latency-throughput tradeoff, including single-request regressions and long-prompt starvation. No single speedup number qualifies a change. Future scheduling should coordinate prefill budget, decode batch size, speculative work and KV pressure with measured queue age. These joint policies are research work, not currently implemented guarantees.

## Current implemented baseline

- Dense Llama/Qwen2 forward, local safetensors loading and QKV/gate-up packing.
- Refcounted KV pages and complete prefix-block reuse.
- Decode-first continuous batching with feedback-sized round-robin prefill.
- Verified greedy prompt lookup with rejected-KV rollback.
- Text API, SSE, readiness, bounded queues, metrics and rotating trace.
- Numerical/state/protocol tests, real Qwen CPU/MPS FP32 gates, benchmark tooling.

The SDPA path still gathers KV, stochastic sampling is on CPU, and scheduling is synchronous. The experimental paged Triton kernel is a development target pending CUDA qualification.

## Next gate CUDA execution and latency

The [first H100 runbook](H100_FIRST_RUN.md) and campaign scripts now provide fixed-revision CPU staging, content verification, bounded subprocess execution, GPU sampling and resumable gates. This is test infrastructure delivered before renting hardware, not evidence that CUDA has passed. The first batch is Qwen2.5 0.5B, 1.5B, 3B and 7B; it does not add the Arena20 architectures.

Choose one GPU and one supported 0.5B/1.5B/7B model with sufficient memory. First pass real token tests for SDPA and Triton. Measure batch-one TPOT using engine-side per-token timing, not SSE gaps. Profile and quantify gather overhead, projection launches, page-table upload and sampling synchronization.

Candidate engineering changes: persistent device page tables; preallocated buffers; split-K decode with a stable reduction; fused RoPE/KV writes; device sampling; fixed-shape CUDA Graph replay. Each requires an ablation and independent correctness check. Existing frameworks already employ many of these methods, so implementing them alone is not a novelty claim.

## Next gate a latency aware policy with demonstrated benefit

Replace fixed draft depth with per-workload cost feedback: proposed/accepted length, target verify cost, fallback cost, context length and waiting requests. Compare against draft-off and fixed-K across repetition and non-repetition cases. Include the cost of finding a draft, verification, rejection and host scheduling.

Explore a joint policy for prefill chunk size and speculative depth under a measured latency target. The current controller only sizes prompt chunks; it does not jointly optimize K or guarantee deadlines. This is a concrete research hypothesis, not a completed algorithm.

## Next gate precision and models

Treat reduced-precision changes as a new execution contract. Investigate MPS FP16 near-tie divergence before claiming exact-output equivalence. Quantization needs weight format loaders, scale/zero-point semantics, matching kernels and quality evaluation, not just casting a tensor.

Add QK Norm and scaled RoPE with reference tests, then selected newer dense architectures. MoE requires routing correctness, expert layout, active/total parameter accounting and relevant kernels. MLA/hybrid attention and multimodal encoders need separate KV/position adapters. Do not advertise universal compatibility based on `config.json` parsing.

## Next gate reliability and platform integration

Run sustained mixed load, cancellation storms, OOM faults, malformed checkpoints, queue saturation, shutdown races and memory-leak tests. Implement a native Model Workbench adapter only after its environment/deployment/readiness/unload contract is defined. The existing OpenAI-compatible endpoint can already serve as an external service, but that is not a complete native integration.

Authentication, secure remote deployment, isolation, observability retention, reproducible release builds and operational SLOs precede production adoption. Publish a narrow research pre-release first, not a “fastest universal production engine” announcement.
