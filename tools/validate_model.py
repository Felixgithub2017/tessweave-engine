"""Real local weights: numerical, generated-token, prefix and batch gates.

Run as PYTHONPATH=. python tools/validate_model.py MODEL --output validation.json.
No downloads, model uploads or remote-code execution. Reports do not contain raw
model tensors. This is a correctness gate, not a throughput benchmark.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from flow_engine.model import CausalLM
from flow_engine.engine import Engine
from flow_engine.config import EngineConfig
from flow_engine.cache import KVPool


def run(engine, requests):
    while engine.step():
        for r in requests:
            # Drain the bounded output mailbox just like a consuming client.
            while not r.events.empty():
                r.events.get_nowait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--attention", choices=["sdpa", "triton"], default="sdpa")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--draft-tokens", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.tokens < 1 or args.threads < 1:
        p.error("positive tokens/threads required")
    output = Path(args.output)
    if output.exists():
        p.error("output exists; use a fresh run filename")
    torch.set_num_threads(args.threads)
    path = Path(args.model).resolve(strict=True)
    dtype = getattr(torch, args.dtype)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            p.error("CUDA requested but unavailable; no fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    prompts = ["用一句话解释什么是大模型推理。", "Explain why caching attention keys and values helps autoregressive inference."]
    ids = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True) for t in prompts]
    started = time.time()
    hf = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                             torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
    refs = []
    with torch.inference_mode():
        ref_logits = hf(torch.tensor([ids[0]], device=args.device)).logits[:, -1].float().cpu()
        for prompt in ids:
            inp = torch.tensor([prompt], device=args.device)
            # Direct HF forward is intentional: generate() may inherit checkpoint
            # repetition penalties or fallback EOS IDs even when passed None.
            # Fixed-length argmax + KV avoids comparing different processors.
            past, generated = None, []
            for _ in range(args.tokens):
                out = hf(inp, past_key_values=past, use_cache=True)
                next_id = out.logits[:, -1].argmax(-1)
                generated.append(int(next_id.item()))
                past = out.past_key_values
                inp = next_id[:, None]
            refs.append(generated)
    reference_peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    # The final output/past hold GPU KV tensors independently of hf. Release
    # them before loading our model; otherwise the reference leaks into our bill.
    del hf, out, past, inp, next_id
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif args.device == "mps":
        torch.mps.empty_cache()
    model = CausalLM.load(path, args.device, dtype)
    pool = KVPool(model.config, 32, 16, dtype, args.device)
    table = pool.allocate((len(ids[0]) + 15) // 16)
    with torch.inference_mode():
        logits = model(torch.tensor([ids[0]], device=args.device), torch.arange(len(ids[0]), device=args.device)[None],
                       [table], [len(ids[0])], pool, args.attention).cpu()
    abs_diff = (logits - ref_logits).abs()
    tolerance = 2e-3 if dtype == torch.float32 else .3
    logit_close = bool(torch.allclose(logits, ref_logits, atol=tolerance, rtol=1e-3 if dtype == torch.float32 else .03))
    del pool
    e = Engine(model, EngineConfig(block_size=16, kv_blocks=64, max_context=512,
                                   prefill_chunk=16, min_prefill_chunk=4, attention=args.attention, draft_tokens=args.draft_tokens))
    cold = []
    for prompt in ids:
        r = e.submit(prompt, args.tokens, ignore_eos=True)
        run(e, [r])
        cold.append(r.output)
    batch = [e.submit(prompt, args.tokens, ignore_eos=True) for prompt in ids]
    run(e, batch)
    matches = [cold[i] == refs[i] == batch[i].output for i in range(len(ids))]
    report = {"model_directory_name": path.name, "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
              "device": args.device, "dtype": args.dtype, "attention": args.attention, "draft_tokens": args.draft_tokens,
              "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(),
              "checked_at_unix": started, "elapsed_s": time.time() - started,
              "reference_method": "HF cached forward + fixed-length argmax; no logits processors or EOS early stop",
              "max_logit_absolute_error": float(abs_diff.max()), "logits_close": logit_close,
              "generated_tokens_per_case": args.tokens, "all_token_ids_equal": all(matches),
              "cases": [{"prompt": prompts[i], "prompt_tokens": len(ids[i]), "token_ids": cold[i],
                         "reference_token_ids": refs[i], "batch_token_ids": batch[i].output,
                         "text": tok.decode(cold[i], skip_special_tokens=True), "matches_reference_and_batch": matches[i],
                         "warm_prefix_tokens": batch[i].cached} for i in range(len(ids))],
              "stats": e.stats(), "scope": "Correctness only; not a GPU speedup or production-readiness claim"}
    if args.device == "cuda":
        torch.cuda.synchronize()
        report["gpu_memory_bytes"] = {"reference_peak_allocated": reference_peak,
                                      "flow_peak_allocated": torch.cuda.max_memory_allocated(),
                                      "flow_peak_reserved": torch.cuda.max_memory_reserved()}
        report["gpu"] = torch.cuda.get_device_name()
    e.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("logits_close", "max_logit_absolute_error", "all_token_ids_equal", "elapsed_s")}, indent=2))
    if not (logit_close and all(matches)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
