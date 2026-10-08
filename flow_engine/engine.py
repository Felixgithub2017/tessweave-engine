from collections import deque
from dataclasses import dataclass, field
import math
import queue
import threading
import time
import uuid
import torch
from .cache import KVPool, CapacityError
from .config import EngineConfig
from .trace import Trace
from .draft import propose


@dataclass
class Request:
    id: str
    prompt: list[int]
    max_tokens: int
    temperature: float
    generator: torch.Generator
    namespace: str
    ignore_eos: bool = False
    tokens: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)
    table: list[int] = field(default_factory=list)
    computed: int = 0
    cached: int = 0
    status: str = "waiting"
    created: float = field(default_factory=time.perf_counter)
    first_token: float | None = None
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=256))


class Engine:
    """Single model worker owns all KV mutations; API clients never run kernels.

    Each turn decodes all ready requests, then advances one prefill round-robin.
    New requests may join on every turn. Worst-case KV reservation at admission
    guarantees admitted requests can finish within their declared token budgets.
    """
    def __init__(self, model, config=None, trace=None):
        self.model = model.eval()
        self.config = config or EngineConfig()
        if self.config.max_context > model.config.max_position_embeddings:
            raise ValueError("Requested context exceeds model configuration")
        weight = next(model.parameters())
        self.device = weight.device
        if self.config.attention == "triton" and self.device.type != "cuda":
            raise ValueError("Triton attention is CUDA-only; select sdpa for CPU/MPS")
        if self.config.attention == "triton":
            from .kernels.paged_decode import paged_decode  # validate optional dependency now
        self.pool = KVPool(model.config, self.config.kv_blocks, self.config.block_size,
                           weight.dtype, weight.device, self.config.prefix_cache)
        self.trace = trace or Trace()
        self.waiting = deque()
        self.active = []
        self.requests = {}
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.running = False
        self.thread = None
        self.fatal = None
        self.prefill_cursor = 0
        self.chunk = self.config.prefill_chunk
        self.prefill_ms_per_token = None
        self.counters = {"submitted": 0, "finished": 0, "cancelled": 0, "failed": 0,
                         "generated_tokens": 0, "prefill_tokens": 0, "steps": 0,
                         "draft_proposed": 0, "draft_accepted": 0, "speculative_steps": 0}

    def submit(self, tokens, max_tokens=128, temperature=0.0, seed=0, namespace="local", ignore_eos=False):
        with self.lock:
            if self.fatal:
                raise RuntimeError(f"Engine failed: {self.fatal}")
            if not tokens or any(type(t) is not int or t < 0 or t >= self.model.config.vocab_size for t in tokens):
                raise ValueError("Prompt must contain valid token IDs")
            if max_tokens < 1 or len(tokens) + max_tokens > self.config.max_context:
                raise ValueError("prompt_tokens + max_tokens exceeds configured context")
            if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                raise ValueError("temperature must be finite and between 0 and 2")
            if math.ceil((len(tokens) + max_tokens) / self.config.block_size) > self.pool.capacity:
                raise ValueError("This request alone exceeds the KV pool; increase --kv-blocks or reduce tokens")
            if len(self.waiting) >= self.config.max_queue:
                raise CapacityError("Waiting queue is full")
            r = Request(uuid.uuid4().hex, list(tokens), max_tokens, temperature,
                        torch.Generator(device="cpu").manual_seed(seed), namespace, ignore_eos)
            r.tokens = list(tokens)
            self.requests[r.id] = r
            self.waiting.append(r)
            self.counters["submitted"] += 1
            self.trace.emit("submit", request_id=r.id, prompt_tokens=len(tokens), max_tokens=max_tokens,
                            temperature=temperature, seed=seed)
            self.wake.set()
            return r

    def _admit(self):
        while self.waiting and len(self.active) < self.config.max_active:
            r = self.waiting[0]
            shared = self.pool.acquire_prefix(r.prompt, r.namespace)
            total = math.ceil((len(r.prompt) + r.max_tokens) / self.config.block_size)
            try:
                extra = self.pool.allocate(total - len(shared))
            except CapacityError:
                self.pool.release(shared)
                break  # FIFO admission; avoid indefinite starvation of a large job.
            self.waiting.popleft()
            r.table = shared + extra
            r.computed = r.cached = len(shared) * self.config.block_size
            self.pool.hits += r.cached
            r.status = "active"
            self.active.append(r)
            self.trace.emit("admit", request_id=r.id, reused_tokens=r.cached,
                            reserved_blocks=total, shared_blocks=len(shared), **self.pool.stats())

    def _emit(self, r, event):
        try:
            r.events.put_nowait(event)
            return True
        except queue.Full:
            self._finish(r, "error", "Consumer too slow; bounded output queue exhausted")
            return False

    def _finish(self, r, reason, error=None):
        if r.id not in self.requests:
            return
        if r in self.active:
            if reason not in {"error", "cancelled"}:
                self.pool.publish(r.tokens, r.computed, r.table, r.namespace)
            self.active.remove(r)
        elif r in self.waiting:
            self.waiting.remove(r)
        self.pool.release(r.table)
        r.table = []
        r.status = reason
        del self.requests[r.id]
        self.counters[{"cancelled": "cancelled", "error": "failed"}.get(reason, "finished")] += 1
        end = {"type": "done", "finish_reason": reason, "error": error,
               "usage": {"prompt_tokens": len(r.prompt), "completion_tokens": len(r.output),
                         "total_tokens": len(r.prompt) + len(r.output)},
               "cached_tokens": r.cached, "elapsed_s": time.perf_counter() - r.created}
        # Ensure a terminal event even when a disconnected client did not drain.
        if r.events.full():
            while not r.events.empty():
                r.events.get_nowait()
        r.events.put_nowait(end)
        self.trace.emit("finish", request_id=r.id, **{k: v for k, v in end.items() if k != "type"})

    def cancel(self, request_id):
        with self.lock:
            r = self.requests.get(request_id)
            if r:
                self._finish(r, "cancelled")

    def _sample(self, r, logits):
        if r.temperature == 0:
            return int(logits.argmax().item())
        probs = torch.softmax(logits.float().cpu() / r.temperature, -1)
        return int(torch.multinomial(probs, 1, generator=r.generator).item())

    def _commit(self, r, token):
        r.tokens.append(token)
        r.output.append(token)
        self.counters["generated_tokens"] += 1
        if r.first_token is None:
            r.first_token = time.perf_counter()
            self.trace.emit("first_token", request_id=r.id, ttft_ms=(r.first_token - r.created) * 1000)
        if not self._emit(r, {"type": "token", "token_id": token}):
            return False
        if not r.ignore_eos and token in self.model.config.eos_token_ids:
            self._finish(r, "stop")
            return False
        if len(r.output) >= r.max_tokens:
            self._finish(r, "length")
            return False
        return True

    def _speculate(self, r):
        """Target verifies a linear draft in one causal pass, greedy only.

        Rejected KV is left physically allocated but outside computed length;
        future writes overwrite it. No rejected token reaches prefix publication.
        """
        maximum = min(self.config.draft_tokens, r.max_tokens - len(r.output) - 1)
        draft = propose(r.tokens, maximum, self.config.draft_window)
        if not draft:
            return None
        started = time.perf_counter()
        old = r.computed
        inputs = [r.tokens[old]] + draft
        positions = torch.arange(old, old + len(inputs), device=self.device)[None]
        logits = self.model(torch.tensor([inputs], device=self.device), positions, [r.table],
                            [old + len(inputs)], self.pool, self.config.attention, all_logits=True)
        target = logits[0].argmax(-1).tolist()
        accepted = 0
        while accepted < len(draft) and draft[accepted] == target[accepted]:
            accepted += 1
        committed = draft[:accepted] + [target[accepted]]
        used = 0
        for token in committed:
            # Advance only the verified prefix, not the entire speculative pass.
            r.computed += 1
            used += 1
            if not self._commit(r, token):
                break
        actual_accepted = min(accepted, used)
        self.counters["draft_proposed"] += len(draft)
        self.counters["draft_accepted"] += actual_accepted
        self.counters["speculative_steps"] += 1
        self.counters["steps"] += 1
        elapsed = (time.perf_counter() - started) * 1000
        self.trace.emit("step", phase="verify", request_ids=[r.id], tokens=len(inputs), committed_tokens=used,
                        proposed=len(draft), accepted=actual_accepted, elapsed_ms=elapsed, **self.pool.stats())
        return elapsed

    def _run(self, requests, count, phase):
        start = time.perf_counter()
        ids = torch.tensor([r.tokens[r.computed:r.computed + count] for r in requests], device=self.device)
        positions = torch.tensor([list(range(r.computed, r.computed + count)) for r in requests], device=self.device)
        lengths = [r.computed + count for r in requests]
        logits = self.model(ids, positions, [r.table for r in requests], lengths, self.pool, self.config.attention)
        # Greedy requests transfer ONE selected ID, not a full vocabulary of
        # logits. Stochastic sampling currently uses a seeded CPU generator.
        sampled = [self._sample(r, row) if r.computed + count >= len(r.prompt) else None
                   for r, row in zip(requests, logits)]
        if all(token is None for token in sampled):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elif self.device.type == "mps":
                torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        self.counters["steps"] += 1
        for r, token in zip(requests, sampled):
            r.computed += count
            if r.computed < len(r.prompt):
                continue
            assert token is not None
            self._commit(r, token)
        self.trace.emit("step", phase=phase, request_ids=[r.id for r in requests],
                        batch=len(requests), tokens=count * len(requests), elapsed_ms=elapsed,
                        prefill_chunk=self.chunk, **self.pool.stats())
        return elapsed

    @torch.inference_mode()
    def step(self):
        with self.lock:
            self._admit()
            if not self.active:
                return False
            decode = [r for r in self.active if r.computed >= len(r.prompt)]
            decode_ms = None
            if len(self.active) == 1 and decode and self.config.draft_tokens and decode[0].temperature == 0:
                decode_ms = self._speculate(decode[0])
            if decode_ms is None:
                decode_ms = self._run(decode, 1, "decode") if decode else 0.0
            prefill = [r for r in self.active if r.computed < len(r.prompt)]
            if prefill:
                r = prefill[self.prefill_cursor % len(prefill)]
                self.prefill_cursor += 1
                count = min(self.chunk, len(r.prompt) - r.computed)
                elapsed = self._run([r], count, "prefill")
                self.counters["prefill_tokens"] += count
                sample = elapsed / count
                self.prefill_ms_per_token = sample if self.prefill_ms_per_token is None else .8 * self.prefill_ms_per_token + .2 * sample
                # Feedback controller, not a hard latency SLA. At least one
                # minimum chunk progresses so long prompts cannot starve.
                budget = max(1.0, self.config.target_step_ms - decode_ms)
                estimate = int(budget / max(.001, self.prefill_ms_per_token))
                self.chunk = max(self.config.min_prefill_chunk, min(self.config.prefill_chunk, estimate))
            return True

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="flow-model-worker", daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                if not self.step():
                    self.wake.wait(.05)
                    self.wake.clear()
            except Exception as exc:
                with self.lock:
                    self.fatal = f"{type(exc).__name__}: {exc}"
                    self.trace.emit("fatal", error=self.fatal)
                    for r in list(self.requests.values()):
                        self._finish(r, "error", self.fatal)
                self.running = False

    def close(self):
        self.running = False
        self.wake.set()
        if self.thread:
            self.thread.join()  # never free storage while kernels are in flight
        with self.lock:
            for r in list(self.requests.values()):
                self._finish(r, "cancelled")
            self.pool.clear_prefixes()
        self.trace.close()

    def stats(self):
        with self.lock:
            return {**self.counters, **self.pool.stats(), "active": len(self.active), "waiting": len(self.waiting),
                    "prefill_chunk": self.chunk, "ready": self.running and not self.fatal, "fatal": self.fatal,
                    "device": str(self.device), "attention": self.config.attention}
