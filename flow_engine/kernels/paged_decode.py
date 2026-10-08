"""Experimental one-query paged attention, online FP32 softmax, native GQA.

One program per request/query head, streaming over physical pages. This avoids
the SDPA reference path's dense KV gather and repeated GQA K/V materialization.
It is not a port of the vLLM/SGLang kernels. CUDA correctness/performance gates
must pass on the target device before this opt-in backend is used.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, TABLE, LENGTH, OUT,
            HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            PAGE: tl.constexpr, TABLE_WIDTH: tl.constexpr,
            BD: tl.constexpr, TILE: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // (HQ // HK)
    length = tl.load(LENGTH + batch)
    d = tl.arange(0, BD)
    q = tl.load(Q + (batch * HQ + head) * D + d, d < D, 0).to(tl.float32)
    acc = tl.full((BD,), 0, tl.float32)
    denom = tl.full((), 0, tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    for start in range(0, tl.cdiv(length, TILE)):
        p = start * TILE + tl.arange(0, TILE)
        block = tl.load(TABLE + batch * TABLE_WIDTH + p // PAGE, p < length, 0)
        slot = block * PAGE + p % PAGE
        offsets = (slot[:, None] * HK + kv_head) * D + d[None, :]
        mask = (p[:, None] < length) & (d[None, :] < D)
        k = tl.load(K + offsets, mask, 0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1) * (D ** -0.5)
        score = tl.where(p < length, score, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(score, axis=0))
        alpha = tl.exp(maximum - new_max)
        prob = tl.exp(score - new_max)
        v = tl.load(V + offsets, mask, 0).to(tl.float32)
        acc = acc * alpha + tl.sum(prob[:, None] * v, axis=0)
        denom = denom * alpha + tl.sum(prob, axis=0)
        maximum = new_max
    tl.store(OUT + (batch * HQ + head) * D + d, acc / denom, d < D)


def paged_decode(q, k, v, tables, lengths):
    if not q.is_cuda or q.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Triton decode requires CUDA and floating point Q/K/V")
    # Validate host metadata before launching a kernel with raw page pointers.
    # Bad page IDs must produce a Python error, not an illegal GPU memory read.
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Q must be B,H,D; K/V must be pages,page_size,KV_heads,D")
    if k.device != q.device or v.device != q.device or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q/K/V must share device and dtype")
    b, heads, dim = q.shape
    if min(b, heads, dim, *k.shape) < 1 or dim != k.shape[-1] or dim > 256 or heads % k.shape[2]:
        raise ValueError("Unsupported Triton attention dimensions")
    if len(tables) != b or len(lengths) != b:
        raise ValueError("One page table and sequence length per request required")
    for table, length in zip(tables, lengths):
        if type(length) is not int or length < 1 or len(table) * k.shape[1] < length:
            raise ValueError("Sequence length exceeds the supplied page table")
        if any(type(page) is not int or not 0 <= page < k.shape[0] for page in table):
            raise ValueError("Physical page ID out of bounds")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    width = max(len(t) for t in tables)
    table = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32, device=q.device)
    lens = torch.tensor(lengths, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    _decode[(b, heads)](q, k, v, table, lens, out, heads, k.shape[2], dim, k.shape[1], width,
                       triton.next_power_of_2(dim), 32, num_warps=4)
    return out
