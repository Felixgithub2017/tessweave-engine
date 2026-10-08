"""Fail-closed CUDA gate: real paged decode against dense attention, no weights."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("Use a fresh output path")
    report = {"status": "failed", "cases": [], "scope": "Synthetic CUDA kernel correctness, not real-model qualification"}
    started = time.time()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: this gate cannot be skipped or passed on CPU/MPS")
        from flow_engine.kernels.paged_decode import paged_decode
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
        torch.manual_seed(20261008)
        torch.backends.cuda.matmul.allow_tf32 = False
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for dim in (32, 64, 128, 256):
                for heads, kv_heads in ((8, 8), (8, 2), (8, 1)):
                    lengths, page = [1, 17, 65, 257], 16
                    # Non-contiguous physical pages and ragged lengths test addressing.
                    tables, used = [], 0
                    permutation = torch.randperm(80).tolist()
                    for length in lengths:
                        count = (length + page - 1) // page
                        tables.append(permutation[used:used + count])
                        used += count
                    k = torch.randn(80, page, kv_heads, dim, device="cuda", dtype=dtype)
                    v = torch.randn_like(k)
                    q = torch.randn(4, heads, dim, device="cuda", dtype=dtype)
                    with torch.inference_mode():
                        actual = paged_decode(q, k, v, tables, lengths)
                        refs = []
                        for row, length in enumerate(lengths):
                            kk = k[tables[row]].reshape(-1, kv_heads, dim)[:length].float()
                            vv = v[tables[row]].reshape(-1, kv_heads, dim)[:length].float()
                            kk = kk.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
                            vv = vv.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
                            score = torch.einsum("hd,hld->hl", q[row].float(), kk) / dim**.5
                            refs.append(torch.einsum("hl,hld->hd", score.softmax(-1), vv))
                        expected = torch.stack(refs)
                        atol = 3e-5 if dtype == torch.float32 else .003 if dtype == torch.float16 else .025
                        error = float((actual.float() - expected).abs().max())
                        passed = bool(torch.allclose(actual.float(), expected, atol=atol, rtol=atol))
                    report["cases"].append({"dtype": str(dtype), "head_dim": dim, "heads": heads,
                                            "kv_heads": kv_heads, "lengths": lengths, "atol": atol,
                                            "max_absolute_error": error, "passed": passed})
                    if not passed:
                        raise AssertionError(f"Paged kernel mismatch: {report['cases'][-1]}")
        torch.cuda.synchronize()
        report["status"] = "passed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_s"] = time.time() - started
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("x") as f:
            json.dump(report, f, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
