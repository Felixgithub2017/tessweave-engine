import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flow_engine.qualification import (LIMIT, MetadataClient, MetadataRedirect,
    _permitted_url, audit_captured, audit_config, audit_remote, load_catalog, main)


class Response(io.BytesIO):
    status = 200
    headers = {}

    def geturl(self):
        return "https://huggingface.co/api/models/test/model"


class QualificationTests(unittest.TestCase):
    def test_frozen_scope_and_deduplication(self):
        catalog = load_catalog()
        self.assertEqual(catalog["leaderboard_date"], "2026-10-02")
        self.assertEqual(len(catalog["models"]), 20)
        self.assertEqual(catalog["models"][0]["repo_id"], "moonshotai/Kimi-K3")
        repos = {m["repo_id"] for m in catalog["models"]}
        self.assertIn("deepseek-ai/DeepSeek-V4-Pro-0813", repos)
        self.assertIn("deepseek-ai/DeepSeek-V4-Pro", repos)
        self.assertNotIn("deepseek-v4-pro-high-preview", {m["arena_model"] for m in catalog["models"]})

    def test_nested_architecture_is_not_silently_dense(self):
        report = audit_config({"model_type": "new_multimodal", "text_config": {
            "model_type": "moe", "num_experts": 256, "num_experts_per_tok": 8,
            "kv_lora_rank": 512, "layer_types": ["linear_attention", "full_attention"],
            "quantization_config": {"quant_method": "fp8"}}, "vision_config": {"hidden_size": 1152}})
        self.assertFalse(report["supported"])
        self.assertEqual(report["native_config_gate"]["status"], "blocked")
        self.assertEqual(report["sections"]["config.text_config"]["num_experts"], 256)
        self.assertIn("moe_routing_and_expert_weights", report["features"])
        self.assertIn("quantized_storage_requires_loader_and_kernel", report["features"])
        self.assertEqual(report["synthetic_forward"], "not_run")

    def test_accepted_config_is_not_inference_qualification(self):
        report = audit_config(dict(model_type="qwen2", hidden_size=32, num_attention_heads=4,
            num_hidden_layers=2, intermediate_size=64, vocab_size=128, max_position_embeddings=128))
        self.assertEqual(report["native_config_gate"]["status"], "config_accepted_only")
        self.assertFalse(report["supported"])
        self.assertEqual(report["real_weights_inference"], "not_run")

    def test_no_weight_or_arbitrary_network_destinations(self):
        for url in ("http://huggingface.co/api/models/x/y", "https://example.com/config.json",
                    "https://huggingface.co/x/y/resolve/main/model.safetensors",
                    "https://huggingface.co/x/y/resolve/main/tokenizer.json",
                    "https://huggingface.co:123/api/models/x/y",
                    "https://secret@huggingface.co/api/models/x/y"):
            self.assertFalse(_permitted_url(url), url)
            with self.assertRaises(ValueError):
                MetadataClient().get_json(url)
        with self.assertRaises(ValueError):
            MetadataRedirect().redirect_request(None, None, 302, "", {}, "https://cdn.example/model.bin")

    def test_bounded_json_and_failure_log(self):
        client = MetadataClient()
        for raw in (b"x" * (LIMIT + 1), b"[]", b"invalid", b"\xff"):
            with patch.object(client.opener, "open", return_value=Response(raw)):
                with self.assertRaises((ValueError, UnicodeError)):
                    client.get_json("https://huggingface.co/api/models/test/model")
        self.assertEqual(len(client.events), 4)
        self.assertTrue(all("error" in e for e in client.events))
        self.assertTrue(all(e["proxy"] == "disabled" for e in client.events))

    def test_sha_pinned_fetch_and_license_provenance(self):
        calls = []
        def get(_self, url):
            calls.append(url)
            if "/api/" in url:
                return {"sha": "a" * 40, "cardData": {"license": "custom"},
                        "siblings": [{"rfilename": "model.safetensors"}]}, "meta-hash"
            return {"model_type": "new_moe"}, "config-hash"
        with patch.object(MetadataClient, "get_json", get):
            r = audit_remote({"repo_id": "test/model", "arena_license": "MIT"})
        self.assertIn("/resolve/" + "a" * 40 + "/config.json", calls[1])
        self.assertEqual(r["repository_license"], "custom")
        self.assertEqual(r["arena_license"], "MIT")
        self.assertFalse(r["supported"])

    def test_timeout_and_missing_sha_do_not_become_success(self):
        with patch.object(MetadataClient, "get_json", side_effect=TimeoutError("offline")):
            r = audit_remote({"repo_id": "test/model"})
        self.assertEqual(r["metadata_status"], "unavailable")
        with patch.object(MetadataClient, "get_json", return_value=({}, "hash")) as get:
            r = audit_remote({"repo_id": "test/model"})
        self.assertEqual(get.call_count, 1)
        self.assertEqual(r["metadata_status"], "unavailable")

    def test_offline_default_no_network_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(MetadataClient, "get_json") as get:
            out = Path(folder) / "report.json"
            with patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(["--output", str(out)]), 0)
            get.assert_not_called()
            report = json.loads(out.read_text())
            self.assertEqual(report["summary"]["inference_qualified"], 0)
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                main(["--output", str(out)])

    def test_captured_metadata_preserves_incomplete_provenance(self):
        bundle = {"models": [dict(m, config={"model_type": "custom"},
                    revision="a" * 40, revision_verified_for_config=False) for m in load_catalog()["models"]]}
        with patch.object(MetadataClient, "get_json") as get:
            report = audit_captured(bundle)
        get.assert_not_called()
        self.assertEqual(report["summary"]["configs_audited"], 20)
        self.assertEqual(report["summary"]["inference_qualified"], 0)
        self.assertTrue(all(not r["revision_verified_for_config"] for r in report["models"]))
        self.assertTrue(all(len(r["canonical_config_sha256"]) == 64 for r in report["models"]))
        bundle["models"].pop()
        with self.assertRaises(ValueError):
            audit_captured(bundle)

    def test_architecture_specific_state_markers(self):
        r = audit_config({"model_type": "novel", "hc_mult": 4, "expert_dtype": "fp4",
                          "compress_ratios": [4, 128], "num_nextn_predict_layers": 1})
        self.assertIn("multi_stream_residual_requires_review", r["features"])
        self.assertIn("expert_precision_can_differ_from_attention_precision", r["features"])
        self.assertIn("mtp_weights_present_not_automatic_speculative_support", r["features"])

    def test_published_audit_matches_targets_without_false_qualification(self):
        path = Path(__file__).resolve().parents[1] / "validation" / "arena-architecture-audit-2026-10-08.json"
        report = json.loads(path.read_text())
        self.assertEqual([r["repo_id"] for r in report["models"]],
                         [r["repo_id"] for r in load_catalog()["models"]])
        self.assertEqual(report["summary"]["configs_audited"], 20)
        self.assertEqual(report["summary"]["inference_qualified"], 0)
        for r in report["models"]:
            self.assertFalse(r["supported"])
            self.assertEqual(r["architecture"]["native_config_gate"]["status"], "blocked")
            self.assertEqual(r["architecture"]["real_weights_inference"], "not_run")
            self.assertTrue(r["config_url"].startswith("https://huggingface.co/"))


if __name__ == "__main__":
    unittest.main()
