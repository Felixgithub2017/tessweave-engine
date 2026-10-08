import importlib.util
from pathlib import Path
import unittest
import threading

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "tools/benchmark_http.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def report(self, elapsed):
        return {"manifest": dict(model_digest="a", tokenizer_digest="b", dtype="fp32", hardware="cpu", cache_regime="off"),
                "workload_sha256": "c", "max_tokens": 4,
                "records": [{"case": "x", "repeat": 0, "elapsed_s": elapsed, "text": "hello",
                             "usage": {"completion_tokens": 4}, "finish_reason": "length"}]}

    def test_equal_outputs_allow_paired_comparison(self):
        self.assertEqual(bench.compare(self.report(2), self.report(1))["paired_median_total_latency_speedup_b_over_a"], 2)

    def test_failed_or_different_output_cannot_claim_speedup(self):
        for change in ({"text": "different"}, {"error": "timeout"}, {"usage": {"completion_tokens": 3}}):
            b = self.report(1)
            b["records"][0].update(change)
            self.assertFalse(bench.compare(self.report(2), b)["eligible"])

    def test_different_dtype_or_hardware_not_comparable(self):
        b = self.report(1)
        b["manifest"]["dtype"] = "fp16"
        self.assertFalse(bench.compare(self.report(2), b)["eligible"])

    def test_load_level_must_match(self):
        b = self.report(1)
        b["concurrency"] = 4
        self.assertFalse(bench.compare(self.report(2), b)["eligible"])

    def test_summary_wall_time_and_failures(self):
        rows = [{"elapsed_s": 2, "first_text_s": .5, "usage": {"completion_tokens": 10}},
                {"elapsed_s": 4, "first_text_s": 1, "usage": {"completion_tokens": 20}},
                {"error": "timeout"}]
        result = bench.summarize(rows, 5)
        self.assertEqual(result["output_tokens_per_s"], 6)
        self.assertEqual(result["completed_requests_per_s"], .4)
        self.assertEqual(result["total_p95_s"], 4)
        self.assertEqual(result["failed"], 1)
        self.assertIsNone(bench.summarize([{"error": "no"}], 1)["total_p50_s"])

    def test_requests_actually_overlap(self):
        barrier = threading.Barrier(2, timeout=5)
        def request(messages):
            barrier.wait()
            return {"elapsed_s": 1, "first_text_s": .1, "usage": {"completion_tokens": 2}}
        rows, summary = bench.run_load([{"id": "x", "messages": []}], 4, 2, request)
        self.assertEqual(summary["completed"], 4)
        self.assertEqual([r["repeat"] for r in rows], [0, 1, 2, 3])
