import importlib.util
from pathlib import Path
import unittest

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
