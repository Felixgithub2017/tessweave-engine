import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from stage_models import digest_file, model_dir, safe_file, save_new, validate_plan, verify_model
from run_h100 import Journal, artifacts_valid, stages
import stage_models


class CampaignTests(unittest.TestCase):
    def fixture(self, folder):
        files = []
        for name, data in (("config.json", b"{}"), ("tokenizer_config.json", b"{}"), ("model.safetensors", b"test-only-not-a-model")):
            path = Path(folder) / name
            path.write_bytes(data)
            files.append({"name": name, "size": len(data), "sha256": digest_file(path)})
        return {"schema": 1, "endpoint": "https://huggingface.co", "models": [
            {"repo_id": "test/tiny", "revision": "a" * 40, "files": files}]}

    def test_digest_verifies_all_files_and_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            plan = validate_plan(self.fixture(d))
            self.assertEqual(len(verify_model(d, plan["models"][0])), 3)
            (Path(d) / "config.json").write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_model(d, plan["models"][0])

    def test_git_blob_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "file"
            p.write_bytes(b"hello")
            self.assertEqual(digest_file(p, git_blob=True), hashlib.sha1(b"blob 5\0hello").hexdigest())

    def test_plan_rejects_mutable_revision_missing_digest_duplicate_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            base = self.fixture(d)
            for change in ("revision", "digest", "duplicate", "path"):
                p = copy.deepcopy(base)
                m = p["models"][0]
                if change == "revision":
                    m["revision"] = "main"
                elif change == "digest":
                    m["files"][0].pop("sha256")
                elif change == "duplicate":
                    m["files"].append(m["files"][0])
                else:
                    m["files"][0]["name"] = "../config.json"
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_plan(p)

    def test_no_code_download_allowlist(self):
        self.assertFalse(safe_file("modeling_qwen.py"))
        self.assertFalse(safe_file("pytorch_model.bin"))
        self.assertTrue(safe_file("model-00001-of-00004.safetensors"))
        for p in ("/config.json", "../config.json", "a\\config.json"):
            with self.assertRaises(ValueError):
                safe_file(p)

    def test_symlinks_and_extra_shards_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m = self.fixture(d)["models"][0]
            (Path(d) / "model-00001-of-00001.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                verify_model(d, m)
            config = Path(d) / "config.json"
            config.unlink()
            config.symlink_to(Path(d) / "tokenizer_config.json")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_model(d, m)

    def test_destination_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            (Path(d) / "test--tiny").symlink_to(outside)
            with self.assertRaises(ValueError):
                model_dir(d, {"repo_id": "test/tiny", "revision": "a" * 40})

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "report.json"
            save_new(p, {"first": True})
            with self.assertRaises(FileExistsError):
                save_new(p, {"first": False})

    def test_resume_requires_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            p.write_text("{}")
            entry = {"artifacts": {str(p): digest_file(p)}}
            self.assertTrue(artifacts_valid(entry))
            p.write_text("[]")
            self.assertFalse(artifacts_valid(entry))
            self.assertFalse(artifacts_valid({}))

    def test_journal_success_and_failure_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            j = Journal(d, time.monotonic() + 10)
            j.run("ok", [sys.executable, "-c", "print('hello')"])
            with self.assertRaises(RuntimeError):
                j.run("fail", [sys.executable, "-c", "raise SystemExit(7)"])
            entries = [json.loads(line) for line in (Path(d) / "commands.jsonl").read_text().splitlines()]
            self.assertEqual(entries[-1]["returncode"], 7)
            self.assertIn("hello", (Path(d) / "ok.log").read_text())

    def test_timeout_terminates_child_and_records_exit(self):
        with tempfile.TemporaryDirectory() as d:
            j = Journal(d, time.monotonic() + .2)
            with self.assertRaises(subprocess.TimeoutExpired):
                j.run("timeout", [sys.executable, "-c", "import time; time.sleep(30)"])
            rows = [json.loads(line) for line in (Path(d) / "commands.jsonl").read_text().splitlines()]
            self.assertIsNotNone(rows[-1]["returncode"])

    def test_preview_never_launches_gpu_or_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            plan_path = Path(d) / "plan.json"
            save_new(plan_path, self.fixture(d))
            output = Path(d) / "not-created"
            r = subprocess.run([sys.executable, str(TOOLS / "run_h100.py"), "--plan", str(plan_path),
                                "--models-root", d, "--output", str(output)], capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(json.loads(r.stdout)["downloads"])
            self.assertFalse(output.exists())
            self.assertEqual(len(stages([{"repo_id": "test/tiny"}])), 6)

    def test_metadata_only_plan_pins_api_sha_and_preserves_lfs_digest(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "plan.json"
            siblings = [SimpleNamespace(rfilename=n, size=2, blob_id="b" * 40,
                                       lfs=SimpleNamespace(sha256="c" * 64) if n.endswith("safetensors") else None)
                        for n in ("config.json", "tokenizer_config.json", "model.safetensors", "model.py")]
            api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=siblings))
            argv = ["stage_models.py", "plan", "--repo", "test/tiny", "--output", str(target)]
            with patch.object(sys, "argv", argv), patch.object(stage_models, "hub_client", return_value=(lambda **kw: api, None)):
                stage_models.main()
            plan = validate_plan(json.loads(target.read_text()))
            self.assertEqual(plan["models"][0]["revision"], "a" * 40)
            self.assertEqual(len(plan["models"][0]["files"]), 3)
            self.assertEqual(plan["models"][0]["files"][-1]["sha256"], "c" * 64)

    def test_download_preview_does_not_initialize_network(self):
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d) / "plan.json"
            save_new(plan, self.fixture(d))
            out = Path(d) / "report.json"
            argv = ["stage_models.py", "download", "--plan", str(plan), "--root", d, "--output", str(out)]
            with patch.object(sys, "argv", argv), patch.object(stage_models, "hub_client", side_effect=AssertionError("network")):
                stage_models.main()
            self.assertFalse(out.exists())

    def test_metadata_timeout_writes_failure_not_partial_valid_plan(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "plan.json"
            def timeout(*args, **kwargs):
                raise TimeoutError("test network timeout")
            api = SimpleNamespace(model_info=timeout)
            argv = ["stage_models.py", "plan", "--repo", "test/tiny", "--output", str(out)]
            with patch.object(sys, "argv", argv), patch.object(stage_models, "hub_client", return_value=(lambda **kw: api, None)):
                with self.assertRaises(TimeoutError):
                    stage_models.main()
            self.assertFalse(out.exists())
            self.assertEqual(json.loads(Path(str(out) + ".failure.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
