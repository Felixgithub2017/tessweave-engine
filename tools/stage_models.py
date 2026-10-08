"""Public HF checkpoints: metadata plan -> explicit download -> offline hash gate.

No weights are fetched by `plan`. No remote Python, credentials, proxies or
model execution. Keep the local_dir cache to resume interrupted Hub downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

DEFAULT_REPOS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                 "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
ENDPOINTS = {"hf": "https://huggingface.co", "hf-mirror": "https://hf-mirror.com"}
SIDECARS = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model",
            "added_tokens.json", "chat_template.jinja", "model.safetensors.index.json",
            "README.md", "LICENSE", "LICENSE.txt"}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def digest_file(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_file(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or len(p.parts) != 1:
        raise ValueError(f"Only root-level checkpoint files are allowed: {name}")
    return name in SIDECARS or bool(re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name))


def validate_plan(plan):
    if plan.get("schema") != 1 or plan.get("endpoint") not in ENDPOINTS.values():
        raise ValueError("Unsupported plan schema/endpoint")
    if not plan.get("models"):
        raise ValueError("Empty plan")
    seen = set()
    for m in plan["models"]:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", m["repo_id"]):
            raise ValueError("Invalid repository ID")
        if m["repo_id"] in seen or not re.fullmatch(r"[0-9a-f]{40}", m["revision"]):
            raise ValueError("Duplicate repo or non-immutable revision")
        seen.add(m["repo_id"])
        names = [f["name"] for f in m["files"]]
        if len(names) != len(set(names)) or "config.json" not in names or "tokenizer_config.json" not in names:
            raise ValueError("Duplicate files or missing config/tokenizer metadata")
        if not any(n.endswith(".safetensors") for n in names):
            raise ValueError("Safetensors weights required")
        for f in m["files"]:
            if not safe_file(f["name"]) or type(f["size"]) is not int or f["size"] < 0:
                raise ValueError("Invalid file entry")
            field = "sha256" if f.get("sha256") else "git_blob_sha1"
            if not re.fullmatch(r"[0-9a-f]{64}" if field == "sha256" else r"[0-9a-f]{40}", f.get(field, "")):
                raise ValueError("Every file needs a publisher content digest")
    return plan


def model_dir(root, model):
    root = Path(root).resolve()
    dest = root / model["repo_id"].replace("/", "--") / model["revision"]
    if not dest.resolve().is_relative_to(root):
        raise ValueError("Destination escapes models root")
    return dest


def verify_model(folder, model):
    folder = Path(folder).resolve()
    rows = []
    for f in model["files"]:
        p = folder / f["name"]
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(folder):
            raise ValueError(f"Missing/unsafe file: {f['name']}")
        if p.stat().st_size != f["size"]:
            raise ValueError(f"Size mismatch: {f['name']}")
        sha = digest_file(p)
        expected = f.get("sha256")
        actual = sha if expected else digest_file(p, git_blob=True)
        if actual != (expected or f["git_blob_sha1"]):
            raise ValueError(f"Content digest mismatch: {f['name']}")
        rows.append({"name": f["name"], "size": f["size"], "sha256": sha})
    # An extra shard can change loaders that glob the directory.
    expected_weights = {f["name"] for f in model["files"] if f["name"].endswith(".safetensors")}
    if {p.name for p in folder.glob("*.safetensors")} != expected_weights:
        raise ValueError("Unexpected safetensors files in checkpoint directory")
    index = folder / "model.safetensors.index.json"
    if index.exists():
        mapped = set(json.loads(index.read_text())["weight_map"].values())
        if mapped != expected_weights:
            raise ValueError("Index/shard inventory mismatch")
    return rows


def hub_client():
    # Disable both environment and OS proxy discovery in the HTTP client.
    for k in list(os.environ):
        if k.lower().endswith("_proxy"):
            os.environ.pop(k)
    os.environ.update(HF_HUB_DISABLE_XET="1", HF_HUB_ENABLE_HF_TRANSFER="0",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1")
    import requests
    from huggingface_hub import HfApi, configure_http_backend, snapshot_download
    def factory():
        session = requests.Session()
        session.trust_env = False
        return session
    configure_http_backend(backend_factory=factory)
    return HfApi, snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan", help="Fetch public metadata only; no weights")
    a.add_argument("--repo", action="append", help="Default: four Qwen2.5 checkpoints, 0.5B through 7B")
    a.add_argument("--source", choices=ENDPOINTS, default="hf")
    a.add_argument("--output", required=True)
    for name in ("download", "verify"):
        a = sub.add_parser(name)
        a.add_argument("--plan", required=True)
        a.add_argument("--root", required=True)
        a.add_argument("--output", required=True, help="Fresh report path, even when resuming downloads")
        if name == "download":
            a.add_argument("--execute", action="store_true", help="Explicit approval to download listed bytes")
            a.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("Report exists; use a fresh path")
    started = time.time()
    if args.command == "plan":
        HfApi, _ = hub_client()
        endpoint = ENDPOINTS[args.source]
        api = HfApi(endpoint=endpoint, token=False)
        models = []
        for repo in args.repo or DEFAULT_REPOS:
            try:
                info = api.model_info(repo, files_metadata=True, timeout=30)
            except Exception as exc:
                save_new(str(args.output) + ".failure.json", {
                    "status": "failed", "repo_id": repo, "endpoint": endpoint,
                    "elapsed_s": time.time() - started, "weights_downloaded": False,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "instruction": "Test direct networking on CPU before renting a GPU; no automatic proxy fallback"})
                raise
            files = []
            for f in info.siblings:
                if "/" in f.rfilename or not safe_file(f.rfilename):
                    continue
                lfs = f.lfs
                files.append({"name": f.rfilename, "size": f.size,
                              "sha256": getattr(lfs, "sha256", None) if lfs else None,
                              "git_blob_sha1": f.blob_id})
            models.append({"repo_id": repo, "revision": info.sha, "files": files})
        plan = validate_plan({"schema": 1, "endpoint": endpoint, "models": models,
                              "created_at_unix": started, "proxy_policy": "direct, no credentials"})
        save_new(args.output, plan)
        print(json.dumps({"plan": args.output, "total_bytes": sum(f["size"] for m in models for f in m["files"]),
                          "weights_downloaded": False}, indent=2))
        return
    plan = validate_plan(json.loads(Path(args.plan).read_text()))
    total = sum(f["size"] for m in plan["models"] for f in m["files"])
    if args.command == "download" and not args.execute:
        print(json.dumps({"download_bytes_upper_bound": total, "root": str(Path(args.root).resolve()),
                          "endpoint": plan["endpoint"], "instruction": "Review plan, then add --execute"}, indent=2))
        return
    report = {"plan_sha256": digest_file(args.plan), "started_at_unix": started,
              "command": sys.argv, "status": "running", "models": []}
    try:
        if args.command == "download":
            if not 1 <= args.workers <= 16:
                p.error("workers must be 1..16")
            _, download = hub_client()
            Path(args.root).mkdir(parents=True, exist_ok=True)
        for m in plan["models"]:
            folder = model_dir(args.root, m)
            if args.command == "download":
                for candidate in [folder / f["name"] for f in m["files"]] + [folder / ".cache", folder / ".cache/huggingface"]:
                    if candidate.is_symlink():
                        raise ValueError("Refusing download through an existing symlink")
                remaining = sum(max(0, f["size"] - ((folder / f["name"]).stat().st_size
                                    if (folder / f["name"]).is_file() else 0)) for f in m["files"])
                if shutil.disk_usage(args.root).free < remaining + 2 * 1024**3:
                    raise ValueError("Insufficient free disk; includes 2 GiB temporary headroom")
                # Exact revision + exact names; SDK retains partial files for retry.
                download(m["repo_id"], revision=m["revision"], endpoint=plan["endpoint"],
                         local_dir=folder, allow_patterns=[f["name"] for f in m["files"]],
                         token=False, max_workers=args.workers)
            rows = verify_model(folder, m)
            report["models"].append({"repo_id": m["repo_id"], "revision": m["revision"],
                                     "files": rows, "verified": True})
            print(f"Verified {m['repo_id']} @ {m['revision']}", flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_s"] = time.time() - started
        save_new(args.output, report)


if __name__ == "__main__":
    main()
