# Connect TessWeave Engine and Studio

## Roles and current boundary

TessWeave Studio (Model Workbench) is the local-first **control plane**: select artifacts, explain resource estimates, prepare and review commands, orchestrate existing backends, and record experiments. TessWeave Engine (formerly Flow Inference) is an independent **inference runtime**: schedule requests, allocate KV pages and execute the model.

```text
Engineer → Studio → inspect / plan / confirm → external training or serving backend
               ↘ register existing HTTP endpoint
                 → TessWeave Engine → scheduler → KV pages → model / GPU
                 ← text stream + usage
                 → Studio conversation log / latency observations

Engine can also be used directly from its CLI or HTTP API, without Studio.
```

This is a protocol connection, not a merged runtime. Studio's planning coverage does not certify Engine's model support. Engine does not contain vLLM/SGLang, and Studio's other backends do not run through Engine.

## Step 1 — Start Engine explicitly

Use Engine's tested environment and a supported, already downloaded checkpoint. The first validation target is Qwen2.5-0.5B-Instruct. The following assumes you are in the Engine repository with its dependencies installed:

```bash
python -m flow_engine.cli serve \
  --model /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --device cpu --dtype float32 --attention sdpa \
  --max-context 2048 --max-active 1 \
  --port 8010 --served-model-name tessweave-qwen \
  --log runs/studio-bridge.jsonl
```

CPU FP32 is a correctness starting point, not the fastest option. For Apple MPS or CUDA use Engine's hardware-specific validation instructions before changing device/dtype/attention. Do not blindly load an arbitrary MoE or multimodal checkpoint.

Wait for the server to complete startup, then check both endpoints:

```bash
curl --fail http://127.0.0.1:8010/health
curl --fail http://127.0.0.1:8010/v1/models
```

Expected: health reports readiness; the model list contains `tessweave-qwen`. Connection refused while loading is not proof of a model error; inspect the Engine terminal. A listed model is not by itself end-to-end generation validation.

## Step 2 — Start Studio separately

In the Studio repository, using its control-plane Python:

```bash
python -m workbench serve --open
```

Open the complete local session URL printed by Studio. Never put that session credential in a public issue. There is no requirement to install both projects into one environment.

Go to **Deployments → Connect existing service** and register:

| Field | Value |
| --- | --- |
| Name | TessWeave Engine · Qwen validation |
| URL | `http://127.0.0.1:8010` (no `/v1` suffix) |
| Model ID | `tessweave-qwen` (not the checkpoint path) |
| Capabilities | Text only |

Use **Check API**, then **Open playground**. Set **temperature = 0**, **thinking mode = Auto**, maximum output = 64, and send a short question.

Expected: streamed text appears, terminal reports actual inference, and Studio records a conversation. Temperature 0 selects greedy decoding for repeatable correctness checks; positive temperature uses Engine's seeded sampling path. Verified prompt-lookup acceleration applies only to greedy requests. Auto means Studio omits the vendor-specific `chat_template_kwargs` field; On/Off is not part of Engine's strict API contract.

**Current limitation:** Studio's separate “verify generation” button sends a thinking override that Engine does not accept. Use the above Playground request for this integration, not that button. One-click Engine-specific readiness/lifecycle support remains work to implement; do not describe all Studio controls as compatible.

## Step 3 — Read evidence at the right layer

| Evidence | Location | Interpretation |
| --- | --- | --- |
| Text, application bytes, arrival timing and usage | Studio `.workbench/chat-recordings/` | API-level observations; chunks are not tokens |
| Commands and application events | Studio `.workbench/operations.log` | Studio-owned operations, not every external terminal command |
| Scheduler, KV and per-step events | Engine `runs/studio-bridge.jsonl` | Engine trace; not a hardware profiler |
| Engine numeric counters | Engine `/metrics` with `flow_*` names | Retained compatible metric names; Studio does not yet map these into all native KV/queue cards |
| CPU/GPU host samples | Studio observatory | Host-wide observations; not exclusive usage by this one request |

Engine's separately launched command is **not automatically captured in Studio's task log**. Keep the launching terminal log with the Engine JSONL file. Engine-native traces are not Chrome/PyTorch profiler traces; replay them with `python -m flow_engine.cli replay --log runs/studio-bridge.jsonl`, not Studio's profiler importer.

## Step 4 — Stop deliberately

Stop the Engine terminal with Ctrl+C and wait for process exit before starting another model. An externally registered service is not an owned Studio job, so Studio cannot unload that process. Closing a Studio tab does not stop Engine. Studio's manual registration does not transfer process ownership.

For cloud use, start both processes on the cloud host and use Studio's documented SSH tunnel/workspace workflow, or explicitly forward the Engine port. In all cases the registered loopback URL is interpreted from the **Studio server's host**, not the browser's computer. Do not expose these single-user unauthenticated Engine endpoints directly to the Internet.

## Integration roadmap, not current claims

1. Reviewed Engine environment and launch recipes with explicit process ownership.
2. Backend-specific request capabilities and real generation readiness.
3. Engine metric mapping and a dedicated trace viewer.
4. Same-checkpoint, same-workload performance comparisons with vLLM/SGLang.

All four require tests and measured evidence. A shared logo or compatible HTTP envelope is not a performance result.
