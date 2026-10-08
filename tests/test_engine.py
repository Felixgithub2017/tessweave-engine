import json
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
from flow_engine.config import ModelConfig, EngineConfig
from flow_engine.model import CausalLM
from flow_engine.cache import KVPool, CapacityError
from flow_engine.engine import Engine

torch.set_num_threads(2)


def fixture(kind="qwen2", tied=False):
    cls, model_cls = (Qwen2Config, Qwen2ForCausalLM) if kind == "qwen2" else (LlamaConfig, LlamaForCausalLM)
    c = cls(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            eos_token_id=None, tie_word_embeddings=tied, attention_dropout=0.0)
    c._attn_implementation = "eager"
    torch.manual_seed(41)
    hf = model_cls(c).eval()
    ours = CausalLM(ModelConfig.from_dict(c.to_dict())).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)
    return hf, ours


def engine_for(model, **kw):
    defaults = dict(block_size=4, kv_blocks=32, max_context=64, prefill_chunk=4, min_prefill_chunk=1)
    defaults.update(kw)
    return Engine(model, EngineConfig(**defaults))


def drain(engine):
    for _ in range(500):
        if not engine.step():
            return
    raise AssertionError("Engine did not terminate")


@torch.inference_mode()
def reference(hf, tokens, n):
    ids = torch.tensor([tokens])
    result = []
    for _ in range(n):
        token = int(hf(ids).logits[0, -1].argmax())
        result.append(token)
        ids = torch.cat((ids, torch.tensor([[token]])), 1)
    return result


class ForwardTests(unittest.TestCase):
    def test_qwen_and_llama_chunked_logits(self):
        for kind in ("qwen2", "llama"):
            with self.subTest(kind=kind):
                hf, model = fixture(kind)
                pool = KVPool(model.config, 8, 4, torch.float32, "cpu")
                table = pool.allocate(4)
                ids = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 14, 15]])
                with torch.inference_mode():
                    model(ids[:, :3], torch.arange(3)[None], [table], [3], pool)
                    model(ids[:, 3:7], torch.arange(3, 7)[None], [table], [7], pool)
                    got = model(ids[:, 7:], torch.arange(7, 9)[None], [table], [9], pool)
                    expected = hf(ids).logits[:, -1]
                torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)

    def test_batched_decode_ragged_lengths(self):
        hf, model = fixture()
        e = engine_for(model)
        prompts = [[4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9, 10]]
        requests = [e.submit(p, 7) for p in prompts]
        drain(e)
        for r, p in zip(requests, prompts):
            self.assertEqual(r.output, reference(hf, p, 7))
        e.close()

    def test_safetensor_loader_tied_and_untied(self):
        for tied in (True, False):
            hf, _ = fixture(tied=tied)
            with tempfile.TemporaryDirectory() as folder:
                hf.save_pretrained(folder, safe_serialization=True)
                loaded = CausalLM.load(folder, fuse_linears=False)
                for name, weight in loaded.named_parameters():
                    torch.testing.assert_close(weight, dict(hf.named_parameters())[name], rtol=0, atol=0)

    def test_packed_qkv_gateup_match_reference(self):
        hf, model = fixture()
        model.pack_projections()
        e = engine_for(model)
        r = e.submit([2, 3, 4, 5, 6], 8)
        drain(e)
        self.assertEqual(r.output, reference(hf, r.prompt, 8))
        e.close()

    def test_unsupported_semantics_rejected(self):
        hf, _ = fixture()
        for override in ({"model_type": "gemma4"}, {"use_sliding_window": True},
                         {"rope_scaling": {"rope_type": "yarn"}}, {"quantization_config": {"bits": 4}},
                         {"auto_map": {"AutoModel": "run_me.py"}}):
            with self.assertRaises(ValueError):
                ModelConfig.from_dict({**hf.config.to_dict(), **override})

    def test_context_and_token_validation(self):
        _, model = fixture()
        e = engine_for(model)
        for prompt, count in (([], 1), ([128], 1), ([1], 64), ([1], 0)):
            with self.assertRaises(ValueError):
                e.submit(prompt, count)
        e.close()


class CacheTests(unittest.TestCase):
    def test_shared_prefix_logits_identical(self):
        _, model = fixture()
        e = engine_for(model)
        p = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        first = e.submit(p, 5)
        drain(e)
        count = e.counters["prefill_tokens"]
        second = e.submit(p, 5)
        drain(e)
        self.assertEqual(first.output, second.output)
        self.assertEqual(second.cached, 8)
        self.assertEqual(e.counters["prefill_tokens"] - count, 1)
        e.close()
        self.assertEqual(len(e.pool.free), e.pool.capacity)

    def test_partial_tail_not_shared_or_overwritten(self):
        hf, model = fixture()
        e = engine_for(model)
        first = e.submit([2, 3, 4, 5, 6, 7], 3)
        drain(e)
        r = e.submit([2, 3, 4, 5, 60, 70], 4)
        drain(e)
        self.assertEqual(r.cached, 4)
        self.assertEqual(r.output, reference(hf, r.prompt, 4))
        e.close()

    def test_isolated_namespace(self):
        _, model = fixture()
        e = engine_for(model)
        a = e.submit([2, 3, 4, 5, 6], 2, namespace="tenant-a")
        drain(e)
        b = e.submit(a.prompt, 2, namespace="tenant-b")
        drain(e)
        self.assertEqual(b.cached, 0)
        e.close()

    def test_eviction_preserves_pinned_blocks(self):
        _, model = fixture()
        p = KVPool(model.config, 2, 4, torch.float32, "cpu")
        table = p.allocate(1)
        p.publish([1, 2, 3, 4], 4, table)
        with self.assertRaises(CapacityError):
            p.allocate(2)
        self.assertEqual(p.refs[table[0]], 1)
        p.release(table)
        self.assertEqual(len(p.allocate(2)), 2)

    def test_admission_reserves_maximum_and_recovers(self):
        _, model = fixture()
        e = engine_for(model, kv_blocks=4)
        a = e.submit([1, 2, 3, 4, 5, 6], 6)
        b = e.submit([8, 9, 10, 11, 12, 13], 6)
        e.step()
        self.assertEqual(b.status, "waiting")
        drain(e)
        self.assertEqual(len(a.output), 6)
        self.assertEqual(len(b.output), 6)
        e.close()


class SchedulingTests(unittest.TestCase):
    def test_speculation_eos_cannot_publish_uncommitted_kv(self):
        from dataclasses import replace
        hf, model = fixture()
        prompt = [3, 4, 5, 6, 7]
        expected = reference(hf, prompt, 12)
        end = next(i for i in range(2, len(expected)) if expected[i] not in expected[:i])
        model.config = replace(model.config, eos_token_ids=(expected[end],))
        e = engine_for(model, draft_tokens=4)
        def proposal(tokens, maximum, window):
            i = len(tokens) - len(prompt)
            return expected[i:i + maximum]
        with patch("flow_engine.engine.propose", side_effect=proposal):
            r = e.submit(prompt, 12)
            drain(e)
        self.assertEqual(r.status, "stop")
        self.assertEqual(r.output, expected[:end + 1])
        self.assertEqual(r.computed, len(prompt) + len(r.output) - 1)
        e.close()
        self.assertEqual(len(e.pool.free), e.pool.capacity)

    def test_speculative_acceptance_and_rejection_equal_greedy(self):
        from flow_engine.draft import propose
        self.assertEqual(propose([1, 2, 3, 8, 9, 1, 2, 3], 2), [8, 9])
        for accept in (True, False):
            hf, model = fixture()
            prompt = [3, 4, 5, 6, 7]
            expected = reference(hf, prompt, 12)
            e = engine_for(model, draft_tokens=3)
            def proposal(tokens, maximum, window):
                i = len(tokens) - len(prompt)
                draft = expected[i:i + maximum]
                return draft if accept else [(t + 1) % model.config.vocab_size for t in draft]
            with patch("flow_engine.engine.propose", side_effect=proposal):
                r = e.submit(prompt, 12)
                drain(e)
            self.assertEqual(r.output, expected)
            self.assertGreater(e.counters["speculative_steps"], 0)
            if accept:
                self.assertGreater(e.counters["draft_accepted"], 0)
                self.assertLess(e.counters["steps"], 14)
            else:
                self.assertEqual(e.counters["draft_accepted"], 0)
            # Full prompt+generation prefix publication cannot retain rejected KV.
            follow = e.submit(prompt + expected[:8], 3)
            drain(e)
            self.assertEqual(follow.output, expected[8:11])
            e.close()

    def test_speculation_disabled_for_sampling_and_multiple_requests(self):
        _, model = fixture()
        e = engine_for(model, draft_tokens=4)
        e.submit([1, 2, 3], 3, temperature=.7)
        with patch("flow_engine.engine.propose", side_effect=AssertionError("draft must not run")):
            drain(e)
        e.close()
        e = engine_for(model, draft_tokens=4)
        e.submit([1, 2, 3], 6)
        e.submit([3, 4, 5], 6)
        with patch("flow_engine.engine.propose", side_effect=AssertionError("multiple active requests must not speculate")):
            e.step()
            e.step()
            e.step()
        drain(e)
        e.close()

    def test_cancel_pending_and_active_no_reference_leak(self):
        _, model = fixture()
        e = engine_for(model, max_active=1)
        a = e.submit([1, 2, 3, 4, 5, 6, 7], 8)
        b = e.submit([1, 2], 8)
        e.step()
        e.cancel(b.id)
        e.cancel(a.id)
        self.assertFalse(e.step())
        self.assertEqual(len(e.pool.free), e.pool.capacity)
        e.close()

    def test_sampling_seed_independent_of_other_requests(self):
        _, model = fixture()
        e = engine_for(model)
        a = e.submit([1, 2, 3, 4, 5], 7, temperature=.8, seed=9)
        drain(e)
        e.submit([9, 8, 7, 6], 9, temperature=.5, seed=8)
        b = e.submit(a.prompt, 7, temperature=.8, seed=9)
        drain(e)
        self.assertEqual(a.output, b.output)
        e.close()

    def test_queue_limit_and_slow_consumer(self):
        _, model = fixture()
        e = engine_for(model, max_queue=1)
        r = e.submit([1, 2], 4)
        with self.assertRaises(CapacityError):
            e.submit([1, 2], 4)
        import queue
        r.events = queue.Queue(maxsize=1)
        drain(e)
        self.assertEqual(r.status, "error")
        self.assertEqual(r.events.get()["finish_reason"], "error")
        e.close()

    def test_fatal_forward_error_releases_resources(self):
        _, model = fixture()
        e = engine_for(model)
        r = e.submit([1, 2], 3)
        with patch.object(model, "forward", side_effect=RuntimeError("test kernel failure")):
            e.start()
            e.thread.join(timeout=3)
        self.assertIn("test kernel failure", e.fatal)
        self.assertEqual(r.events.get()["finish_reason"], "error")
        self.assertEqual(len(e.pool.free), e.pool.capacity)
        e.close()


class CUDATests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required; not a passed GPU gate")
    def test_triton_paged_matches_reference(self):
        # Real-device test with non-contiguous pages, GQA and ragged batches.
        from flow_engine.kernels.paged_decode import paged_decode
        from torch.nn import functional as F
        torch.manual_seed(5)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for dim in (32, 64, 128):
                q = torch.randn(2, 8, dim, device="cuda", dtype=dtype)
                k = torch.randn(8, 16, 2, dim, device="cuda", dtype=dtype)
                v = torch.randn_like(k)
                tables, lengths = [[3, 1, 6], [4, 7]], [37, 19]
                got = paged_decode(q, k, v, tables, lengths)
                expected = []
                for i in range(2):
                    keys = k[tables[i]].flatten(0, 1)[:lengths[i]].repeat_interleave(4, 1).transpose(0, 1)[None]
                    vals = v[tables[i]].flatten(0, 1)[:lengths[i]].repeat_interleave(4, 1).transpose(0, 1)[None]
                    expected.append(F.scaled_dot_product_attention(q[i][None, :, None], keys, vals)[0, :, 0])
                torch.testing.assert_close(got, torch.stack(expected), atol=.015, rtol=.02)


if __name__ == "__main__":
    unittest.main()
