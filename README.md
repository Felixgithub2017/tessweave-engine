# Flow Inference

An independent language-model inference engine for inspecting and improving low-concurrency latency. Flow owns the request scheduler, physical KV pages, prefix index, Transformer forward pass, greedy speculative verification and HTTP service. It does **not** delegate inference to vLLM, SGLang or `transformers.generate()`.

**Status: 0.1.0a1, pre-alpha.** Real Qwen2.5-0.5B-Instruct weights have passed CPU and Apple GPU FP32 forward/token checks against Transformers. CUDA kernels, large models and production reliability remain separate release gates. No faster-than-vLLM/SGLang claim is made. Reduced-precision output differences are documented, not hidden.

Local checks include CPU correctness, service, Arena qualification and campaign-safety tests, plus recorded real checkpoint validation. CUDA remains a separate, unpassed gate. Use the source checkout for the latest changes. Hosted CI and public publication are not implied by local test results.

[中文手把手说明](docs/使用说明.md) · [Architecture and state walkthrough](docs/ARCHITECTURE.md) · [Validation evidence](validation/README.md) · [Benchmark protocol](docs/BENCHMARKS.md) · [Roadmap](docs/ROADMAP.md)

## First H100 campaign

[CPU pre-download and H100 runbook](docs/H100_FIRST_RUN.md) separates storage preparation from billed GPU work. Reserve an initial **4-hour engineering window**, excluding pre-staged downloads; this is a budget, not measured runtime. Validate Qwen2.5 0.5B before progressing to 1.5B, 3B and 7B.

* `tools/stage_models.py plan` reads public metadata and freezes checkpoint revisions and content digests. `download` previews by default; `--execute` downloads directly without proxies or credentials. `verify` checks files offline.
* `tools/run_h100.py` previews the campaign by default. Explicit execution records commands/output and GPU samples, enforces a time budget, and stops at the first failed gate. `--resume` requires unchanged code/environment and intact successful results.
* `tools/validate_cuda.py` requires real CUDA and tests 36 paged-attention combinations. CPU/MPS cannot satisfy this gate.

The runner never rents/releases a machine. **Stopping the script does not stop cloud billing.** No H100 result or broad-model support is claimed before these gates execute on that hardware.

## What is actually implemented

| Mechanism | Implementation | Tradeoff |
|---|---|---|
| Physical KV paging | Refcounted per-layer K/V pool and request block tables | Entire worst-case request budget reserved at admission; conservative utilization |
| Shared prefix reuse | Parent-hashed full blocks, LRU cache ownership, immutable shared pages | Not SGLang's radix-tree implementation; no partial-block reuse or disk tier |
| Continuous decode batching | Ready requests join each turn; active requests decode together | Single worker/device; no distributed parallelism or overlap |
| Chunked prefill | Decode first, then one round-robin prompt chunk | Observed timing tunes chunk size; target is not a hard latency SLA |
| Linear packing | Q/K/V and gate/up row concatenation | 7 linear calls become 4 per layer; this does not imply 43% end-to-end speedup |
| Greedy prompt lookup | Bounded history proposes; target verifies; rejected KV rolled back logically | Opt-in, one active request, greedy only; can be slower when proposals fail |
| Portable attention | PyTorch SDPA with offset-aware masks | Gathers KV and repeats GQA heads; a correctness-oriented path, not state-of-the-art bandwidth efficiency |
| Experimental CUDA attention | Independent Triton decode kernel reads paged K/V directly, online FP32 softmax | Opt-in and unvalidated on CUDA in the current evidence set |
| Service and evidence | OpenAI-style text chat, SSE, health, Prometheus metrics, rotating JSONL trace | Loopback only, no authentication/multitenancy, no production availability guarantee |

## Supported models

The safetensors loader accepts **dense Llama and Qwen2/Qwen2.5 text architectures with default RoPE and full attention**. It rejects quantized, MoE, sliding-window, scaled-RoPE and custom-code checkpoints. Llama 3-style scaled RoPE, Qwen3, GLM, DeepSeek, Gemma, MiniMax and multimodal models are **not yet supported**. Architecture family acceptance is not a claim that every family checkpoint has been tested.

The engine only uses Transformers for tokenizer loading. It implements the neural network in [model.py](flow_engine/model.py); tests use Transformers model classes as an independent reference. Model weights are never downloaded implicitly or included in the release.

### LMArena expansion targets

The [20-model target and validation guide](docs/ARENA_TARGETS.md) uses **LMArena Text Overall only**, frozen to the page dated October 2, 2026. Serving aliases are mapped to checkpoint candidates, not assumed to be identical online weights. The [architecture audit](validation/arena-architecture-audit-2026-10-08.json) covers all 20 configs; **none of these 20 is yet qualified for native Flow inference**. This does not broaden the supported-model list above.

```bash
# Standard-library-only audit; no model weights, remote code or implicit network.
python -m flow_engine.qualification
python -m flow_engine.qualification --config /absolute/path/config.json
# Explicit metadata-only retrieval, direct connection without proxies.
python -m flow_engine.qualification --fetch --output runs/arena-audit.json
```

The live audit pins each successful config fetch to a repository commit and records HTTP responses/errors. Existing reports are never overwritten. Configuration checks, meta-shape execution, synthetic numeric tests, real-weight correctness and GPU benchmarks remain separate gates.

## Install with Miniforge or Miniconda

Run inside this repository:

```bash
conda env create -f environment.yml
conda activate flow-inference
flow-inference doctor
python -B -m unittest discover -s tests -v
```

This is a new environment; it does not modify your existing ML environments. For a pinned local reproduction, use `python -m pip install -c requirements-tested.txt -e '.[test]'` in a compatible environment. CUDA users must first install a PyTorch build appropriate for their driver/GPU. The constraints file records a Mac test environment, not a CUDA installer.

For local Qwen2.5 weights:

```bash
flow-inference inspect /absolute/path/Qwen2.5-0.5B-Instruct
flow-inference generate /absolute/path/Qwen2.5-0.5B-Instruct \
  --prompt '用三句话解释 KV Cache。' --max-tokens 96
flow-inference serve /absolute/path/Qwen2.5-0.5B-Instruct \
  --port 8010 --served-model-name flow-model --log runs/server.jsonl
```

`auto` selects CUDA, then Apple MPS, then CPU. CPU/MPS default to FP32, the currently validated path. Explicit `--dtype float16` / `bfloat16` are experiments and may change near-tied greedy decisions. Validate your exact checkpoint, precision and hardware before relying on them.

The API is served at `http://127.0.0.1:8010`. Startup does a real model warmup. The socket is not presented as ready before loading/warmup succeeds.

```bash
curl --noproxy '*' http://127.0.0.1:8010/health
curl --noproxy '*' http://127.0.0.1:8010/v1/models
curl --noproxy '*' -N http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"flow-model","messages":[{"role":"user","content":"Explain KV caching."}],"temperature":0,"max_tokens":96,"stream":true}'
```

Use base URL `http://127.0.0.1:8010/v1` in an OpenAI-compatible client or the existing visualization platform's external-service registration. This does not add a new native engine selector to that platform. Stop with Ctrl+C to unload; restart with a different model path. For a cloud host, use SSH local port forwarding; the engine deliberately offers no unauthenticated public bind option.

## Understand the main controls

| Option | Default | Meaning |
|---|---:|---|
| `--max-context` | 2048 | Prompt plus maximum generation token budget |
| `--kv-blocks` | 256 | Physical K/V blocks preallocated on the execution device |
| `--block-size` | 16 | Tokens per physical block |
| `--max-active` | 4 | Maximum admitted requests sharing the worker |
| `--prefill-chunk` | 256 | Upper bound for one prompt advance |
| `--target-step-ms` | 50 | Feedback budget for a decode-plus-prefill turn; not a guarantee |
| `--no-prefix-cache` | off | Disable prefix reuse for cold mechanism comparisons |
| `--draft-tokens` | 0 | 0 off; 1–8 greedy draft tokens verified in one target forward |
| `--draft-window` | 512 | Maximum committed history searched for repeated suffixes |
| `--unfused` | off | Disable packed QKV/gate-up projections for ablation |
| `--attention` | sdpa | Portable `sdpa` or experimental CUDA `triton` decode |
| `--log` | runs/engine.jsonl | Rotating structured execution trace, up to five 20 MiB files |

The server accepts text `messages`, `max_tokens`, `temperature`, `seed`, `stream`, and `stream_options.include_usage`. Unknown options are rejected rather than silently ignored. No tool calling, images, top-p or repetition penalty in this release. Checkpoint EOS IDs are read, but saved sampling recipes are not implicitly imported.

KV allocation is `2 × layers × KV_heads × head_dim × element_bytes × block_size × kv_blocks`. Weights, transient activations, the SDPA gather and allocator overhead are **additional**. No automatic out-of-memory prevention is promised by this formula. See [the resource walkthrough](docs/ARCHITECTURE.md).

## Verify before optimizing

```bash
PYTHONPATH=. python tools/validate_model.py /absolute/path/model \
  --device cpu --dtype float32 --tokens 64 --draft-tokens 4 \
  --output runs/correctness.json
PYTHONPATH=. python tools/validate_model.py /absolute/path/model \
  --device mps --dtype float32 --tokens 64 --draft-tokens 4 \
  --output runs/correctness-mps.json
```

On a compatible Linux CUDA environment, install `.[cuda]`, run the unit suite (the CUDA test must execute, not skip), then run the real model gate with `--device cuda --attention triton` at your chosen dtype. Failure means that combination is not qualified.

For service comparisons, fill [the manifest](examples/manifest.template.json) and follow [BENCHMARKS.md](docs/BENCHMARKS.md). The supplied workload is a smoke suite, not evidence for a general performance claim. There is no automatic GPU rental or paid service call.

## Follow the execution

```bash
flow-inference replay runs/server.jsonl
curl --noproxy '*' http://127.0.0.1:8010/metrics
```

JSONL events cover launch/configuration, admission/reservation, prefix reuse, prefill/decode/verification steps, first-token timing, completion and errors. Raw prompt text/token IDs are omitted by default. Replay is a chronological event viewer, not a GPU-kernel profiler or a chat-response replay. Benchmark reports intentionally include responses so exact output comparisons are possible.

## Source layout

```text
flow_engine/
  config.py               architecture and engine constraints
  model.py                safetensors loader, packed Transformer forward
  cache.py                physical pages, prefix ownership, eviction
  engine.py               admission, scheduling, verification, cancellation
  draft.py                bounded committed-history proposals
  kernels/paged_decode.py optional CUDA online-softmax kernel
  server.py               text chat protocol, SSE and readiness
  trace.py                rotating structured audit
  cli.py                  inspect, doctor, generate, serve, replay
tools/                    real checkpoint gates and HTTP benchmark
tests/                    reference-model, state-machine and protocol tests
validation/               recorded successes AND failed experiments
```

## Open source release

Original project code uses the [MIT license](LICENSE); dependencies and model licenses remain separate. [NOTICE](NOTICE) and [source references](docs/SOURCES.md) credit the upstream designs. [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [release gates](docs/RELEASE.md) define the contribution and publication workflow.

This folder is ready to become a standalone repository; no public GitHub repository or PyPI publication is claimed until a destination is selected and publishing succeeds. “Flow Inference” is a working name, not a registered package-name availability claim.
