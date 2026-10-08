import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import torch
from .config import ModelConfig, EngineConfig
from .model import CausalLM
from .engine import Engine
from .trace import Trace


def make_engine(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # CPU/MPS default to the validated FP32 path. Reduced precision can change
    # near-tied argmax decisions and is an explicit experiment on Apple GPUs.
    dtype = args.dtype if args.dtype != "auto" else ("float32" if device in {"cpu", "mps"} else
                                                    "bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    if device == "cuda" or dtype != "float32":
        print("Experimental device/precision combination: run tools/validate_model.py before performance claims.", file=sys.stderr)
    trace = Trace(args.log)
    audit_argv = list(sys.argv)
    for i, argument in enumerate(audit_argv):
        if argument == "--prompt" and i + 1 < len(audit_argv):
            audit_argv[i + 1] = "[REDACTED]"
        elif argument.startswith("--prompt="):
            audit_argv[i] = "--prompt=[REDACTED]"
    trace.emit("launch", argv=audit_argv, python=sys.version, torch=torch.__version__, device=device, dtype=dtype)
    print(f"Loading local weights: {args.model} ({device}, {dtype})", file=sys.stderr, flush=True)
    model = CausalLM.load(args.model, device, getattr(torch, dtype), fuse_linears=not args.unfused)
    config = EngineConfig(block_size=args.block_size, kv_blocks=args.kv_blocks, max_context=args.max_context,
                          max_active=args.max_active, prefill_chunk=args.prefill_chunk,
                          min_prefill_chunk=min(16, args.prefill_chunk), target_step_ms=args.target_step_ms,
                          prefix_cache=not args.no_prefix_cache, attention=args.attention,
                          draft_tokens=args.draft_tokens, draft_window=args.draft_window)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("A tokenizer chat_template is required; no template will be guessed")
    engine = Engine(model, config, trace)
    trace.emit("loaded", engine=asdict(config), model=asdict(model.config), kv_bytes=engine.pool.allocated_bytes)
    return engine, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Independent Flow Inference engine (pre-alpha)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only runtime and accelerator report")
    inspect = sub.add_parser("inspect", help="Validate architecture and explain KV allocation")
    inspect.add_argument("model")
    inspect.add_argument("--kv-blocks", type=int, default=256)
    inspect.add_argument("--block-size", type=int, default=16)
    replay = sub.add_parser("replay", help="Summarize JSONL scheduler events (no inference)")
    replay.add_argument("log")
    for cmd in ("serve", "generate"):
        p = sub.add_parser(cmd)
        p.add_argument("model", help="Local HF safetensors directory; no automatic downloads")
        p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
        p.add_argument("--max-context", type=int, default=2048)
        p.add_argument("--kv-blocks", type=int, default=256)
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--max-active", type=int, default=4)
        p.add_argument("--prefill-chunk", type=int, default=256)
        p.add_argument("--target-step-ms", type=float, default=50)
        p.add_argument("--attention", choices=["sdpa", "triton"], default="sdpa")
        p.add_argument("--no-prefix-cache", action="store_true")
        p.add_argument("--draft-tokens", type=int, default=0, help="0 disables; 1..8 enables verified greedy prompt lookup")
        p.add_argument("--draft-window", type=int, default=512)
        p.add_argument("--unfused", action="store_true", help="Ablation: do not pack QKV/gate-up linear projections")
        p.add_argument("--log", default="runs/engine.jsonl")
        if cmd == "serve":
            p.add_argument("--port", type=int, default=8010)
            p.add_argument("--served-model-name", default="flow-model")
        else:
            p.add_argument("--prompt", required=True)
            p.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps({"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                          "cuda": torch.cuda.is_available(), "mps_built": torch.backends.mps.is_built(),
                          "mps_available": torch.backends.mps.is_available()}, indent=2))
    elif args.command == "inspect":
        c = ModelConfig.read(args.model)
        if min(args.kv_blocks, args.block_size) < 1:
            parser.error("Positive block count/size required")
        print(json.dumps({"architecture": asdict(c), "kv_formula": "2 * layers * KV_heads * head_dim * bytes_per_element * blocks * block_size",
                          "kv_fp16_bytes": c.kv_bytes_per_token(2) * args.kv_blocks * args.block_size,
                          "kv_fp32_bytes": c.kv_bytes_per_token(4) * args.kv_blocks * args.block_size,
                          "excludes": ["weights", "prefill activations", "SDPA gather workspace", "runtime/allocator"]}, indent=2))
    elif args.command == "replay":
        for line in Path(args.log).read_text().splitlines():
            event = json.loads(line)
            if event["event"] in {"step", "admit", "finish", "first_token", "fatal"}:
                print(json.dumps(event, ensure_ascii=False))
    else:
        engine, tokenizer = make_engine(args)
        if args.command == "serve":
            import uvicorn
            from .server import create_app
            # No unauthenticated non-loopback binding in this release.
            uvicorn.run(create_app(engine, tokenizer, args.served_model_name), host="127.0.0.1", port=args.port)
        else:
            try:
                ids = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True)
                r = engine.submit(ids, args.max_tokens)
                engine.start()
                while True:
                    event = r.events.get()
                    if event["type"] == "done":
                        if event["error"]:
                            raise RuntimeError(event["error"])
                        break
                print(tokenizer.decode(r.output, skip_special_tokens=True))
                print(json.dumps(event), file=sys.stderr)
            finally:
                engine.close()


if __name__ == "__main__":
    main()
