"""Closed-loop OpenAI-compatible service benchmark; DIRECT network only.

Reports first TEXT latency, NOT first token latency. SSE event gaps are not
inter-token gaps. Failures, empty responses and missing usage invalidate a run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.parse import urlparse


def request_one(url, model, messages, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    opener = build_opener(ProxyHandler({}))
    start = time.perf_counter()
    first, text, usage, finish, done, total, events = None, "", None, None, False, 0, 0
    with opener.open(Request(url.rstrip("/") + "/chat/completions", data=body,
                             headers={"Content-Type": "application/json"}), timeout=timeout) as response:
        for raw in response:
            total += len(raw)
            if total > 64 * 1024 * 1024 or time.perf_counter() - start > timeout:
                raise RuntimeError("Response exceeded byte/time bound")
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            if not payload:
                continue
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}).get("content", "") or ""
                if delta:
                    first = first if first is not None else time.perf_counter() - start
                    text += delta
                    events += 1
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    if not done or not finish or not text or not usage or usage.get("completion_tokens", 0) < 1:
        raise RuntimeError("Incomplete/empty response or missing authoritative token usage")
    return {"first_text_s": first, "elapsed_s": elapsed, "text_events_not_tokens": events,
            "usage": usage, "finish_reason": finish, "text": text,
            "completion_tokens_per_total_second": usage["completion_tokens"] / elapsed}


def compare(a, b):
    invariants = ("model_digest", "tokenizer_digest", "dtype", "hardware", "cache_regime")
    problems = [k for k in invariants if a["manifest"].get(k) != b["manifest"].get(k)]
    if a.get("concurrency", 1) != b.get("concurrency", 1):
        problems.append("concurrency")
    if a["workload_sha256"] != b["workload_sha256"] or a["max_tokens"] != b["max_tokens"]:
        problems.append("workload/max_tokens")
    key = lambda r: (r["case"], r["repeat"])
    ra, rb = {key(r): r for r in a["records"]}, {key(r): r for r in b["records"]}
    if set(ra) != set(rb) or not ra:
        problems.append("unpaired/empty records")
    ratios = []
    for k in ra.keys() & rb.keys():
        x, y = ra[k], rb[k]
        if x.get("error") or y.get("error") or any(x.get(f) != y.get(f) for f in ("text", "usage", "finish_reason")):
            problems.append(f"output/usage/failure at {k}")
        else:
            ratios.append(x["elapsed_s"] / y["elapsed_s"])
    return {"eligible": not problems, "problems": problems,
            "paired_median_total_latency_speedup_b_over_a": statistics.median(ratios) if ratios and not problems else None,
            "meaning": "Same-output wall latency; not pure decode TPS or statistical significance"}


def summarize(records, wall_s):
    """Rates use campaign wall time, not summed overlapping request time."""
    good = [r for r in records if not r.get("error")]
    def percentile(field, q):
        values = sorted(r[field] for r in good)
        return values[max(0, math.ceil(q * len(values)) - 1)] if values else None
    return {"attempted": len(records), "completed": len(good),
            "failed": len(records) - len(good), "wall_s": wall_s,
            "completed_requests_per_s": len(good) / wall_s if wall_s > 0 else None,
            "output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in good) / wall_s if wall_s > 0 else None,
            "first_text_p50_s": percentile("first_text_s", .50),
            "first_text_p95_s": percentile("first_text_s", .95),
            "total_p50_s": percentile("elapsed_s", .50),
            "total_p95_s": percentile("elapsed_s", .95),
            "percentile_method": "nearest rank, successful requests only; inspect failures too"}


def run_load(cases, repeats, concurrency, request):
    def execute(item):
        i, c = item
        try:
            row = request(c["messages"])
        except Exception as exc:
            row = {"error": str(exc)}
        return {"case": c["id"], "repeat": i, **row}
    start = time.perf_counter()
    # At most C HTTP requests in flight; worker completion admits the next case.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(execute, ((i, c) for i in range(repeats) for c in cases)))
    return records, summarize(records, time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    p.add_argument("--model", default="flow-model")
    p.add_argument("--workload", default="examples/workload.jsonl")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=120)
    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()))
        print(json.dumps(result, indent=2))
        if not result["eligible"]:
            raise SystemExit(2)
        return
    if min(args.repeats, args.max_tokens, args.timeout) <= 0:
        parser.error("positive repeats/tokens/timeout required")
    if not 1 <= args.concurrency <= 256:
        parser.error("concurrency must be between 1 and 256")
    if urlparse(args.base_url).scheme not in {"http", "https"}:
        parser.error("HTTP(S) URL required")
    manifest = json.loads(Path(args.manifest).read_text())
    for field in ("model_digest", "tokenizer_digest", "dtype", "hardware", "cache_regime", "engine_version", "launch_command"):
        if not manifest.get(field) or "REPLACE" in str(manifest[field]):
            parser.error(f"Fill manifest field: {field}")
    workload = Path(args.workload).read_bytes()
    cases = [json.loads(line) for line in workload.splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("Nonempty workload with unique case IDs required")
    out = Path(args.output)
    if out.exists():
        parser.error("Use a fresh output path")
    records, summary = run_load(cases, args.repeats, args.concurrency,
                               lambda messages: request_one(args.base_url, args.model, messages, args.max_tokens, args.timeout))
    print(json.dumps(summary, indent=2), flush=True)
    report = {"manifest": manifest, "workload_sha256": hashlib.sha256(workload).hexdigest(),
              "max_tokens": args.max_tokens, "records": records, "concurrency": args.concurrency,
              "load_mode": "closed_loop", "summary": summary}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if any(r.get("error") for r in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
