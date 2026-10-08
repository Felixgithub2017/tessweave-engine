"""Bounded offline single-GPU campaign. Preview by default; no installs/rental.

Run from the repository root. Command output is streamed to terminal and audit
files. Failed gates stop later stages. Resume skips only successful stages with
the same code, model plan, device and settings and intact recorded artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import build_opener, ProxyHandler

from stage_models import digest_file, model_dir, save_new, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def code_digest():
    h = hashlib.sha256()
    for folder in ("flow_engine", "tools", "tests", "examples"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".jsonl"}:
                h.update(str(p.relative_to(ROOT)).encode())
                h.update(p.read_bytes())
    h.update((ROOT / "pyproject.toml").read_bytes())
    return h.hexdigest()


def stop_process(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


class Journal:
    def __init__(self, folder, deadline):
        self.folder, self.deadline = Path(folder), deadline
        self.env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}

    def event(self, **value):
        with (self.folder / "commands.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time_unix": time.time(), **value}) + "\n")

    def launch(self, name, argv):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Campaign time budget exhausted")
        self.event(event="start", stage=name, argv=argv, cwd=str(ROOT))
        proc = subprocess.Popen(argv, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        def copy():
            with (self.folder / f"{name}.log").open("ab") as log:
                for line in iter(proc.stdout.readline, b""):
                    log.write(line)
                    log.flush()
                    print(line.decode("utf-8", errors="replace"), end="", flush=True)
            proc.stdout.close()
        thread = threading.Thread(target=copy, daemon=True)
        thread.start()
        return proc, thread

    def finish(self, name, proc, thread, terminate=False):
        try:
            if terminate:
                stop_process(proc)
            else:
                proc.wait(timeout=max(.01, self.deadline - time.monotonic()))
        except BaseException:
            stop_process(proc)
            raise
        finally:
            thread.join(timeout=15)
            self.event(event="exit", stage=name, returncode=proc.poll(), terminated_by_runner=terminate)
        if not terminate and proc.returncode:
            raise RuntimeError(f"Stage {name} exited {proc.returncode}; inspect {name}.log")

    def run(self, name, argv):
        self.finish(name, *self.launch(name, argv))


def artifacts_valid(stage):
    return bool(stage.get("artifacts")) and all(Path(p).is_file() and digest_file(p) == sha
                                                for p, sha in stage["artifacts"].items())


def wait_ready(proc, base, deadline):
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server exited before readiness")
        try:
            with opener.open(base + "/health", timeout=2) as r:
                if json.load(r).get("ready"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(.5)
    raise TimeoutError("Readiness timeout; inspect server log")


def stages(models):
    result = [("unit", None, None, None), ("cuda-kernel", None, None, None)]
    for m in models:
        name = m["repo_id"].split("/")[-1]
        for dtype, attention in (("float32", "sdpa"), ("bfloat16", "sdpa"), ("bfloat16", "triton")):
            result.append((f"{name}-{dtype}-{attention}", m, dtype, attention))
        result.append((f"{name}-http", m, "bfloat16", "triton"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--models-root", required=True)
    p.add_argument("--output", required=True, help="Campaign directory; --resume to reuse")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", action="append", help="Exact repo ID; repeat to select several")
    p.add_argument("--budget-hours", type=float, default=4)
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; stopping owned child processes")
    signal.signal(signal.SIGTERM, interrupted)
    if not 0 < args.budget_hours <= 24 or not 1024 <= args.port <= 65535:
        p.error("budget must be (0,24] hours; port 1024..65535")
    plan = validate_plan(json.loads(Path(args.plan).read_text()))
    models = [m for m in plan["models"] if not args.only or m["repo_id"] in args.only]
    if not models or (args.only and set(args.only) != {m["repo_id"] for m in models}):
        p.error("Selected models must occur in plan")
    workflow = stages(models)
    if not args.execute:
        print(json.dumps({"stages": [r[0] for r in workflow], "models": [str(model_dir(args.models_root, m)) for m in models],
                          "budget_hours_per_invocation": args.budget_hours, "downloads": False,
                          "scope": "Single CUDA GPU, existing dense adapters only; no Arena20 qualification",
                          "execute": "Review then add --execute; use --only for the 0.5B first gate"}, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        p.error("CUDA unavailable; refusing a CPU/MPS fallback")
    if torch.cuda.device_count() != 1:
        p.error("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 70 * 1024**3:
        p.error("Campaign requires at least 70 GiB visible VRAM")
    if not torch.cuda.is_bf16_supported():
        p.error("BF16 support required")
    folder = Path(args.output).resolve()
    if folder.exists() and not args.resume:
        p.error("Output directory exists; use a new directory or --resume")
    if args.resume and not (folder / "state.json").exists():
        p.error("No campaign state to resume")
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256": code_digest(), "plan_sha256": digest_file(args.plan),
                "models": [m["repo_id"] for m in models], "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": str(device), "python": sys.version, "port": args.port,
                "models_root": str(Path(args.models_root).resolve())}
    # Freeze all package versions and GPU UUID/driver, not transient utilization.
    from importlib.metadata import distributions
    identity["packages"] = sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])
    identity["packages"] = [list(row) for row in identity["packages"]]
    identity["gpu_identity"] = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                                                       "--format=csv,noheader"], text=True, timeout=15)
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text()) if args.resume else {"identity": identity, "stages": {}}
    if state["identity"] != identity:
        p.error("Code/environment/plan/device changed; start a new campaign")
    def persist():
        temporary = folder / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
    journal = Journal(folder, time.monotonic() + args.budget_hours * 3600)
    state["status"] = "running"
    state.pop("error", None)
    started = time.time()
    state.setdefault("invocations", []).append({"started_at_unix": started, "budget_hours": args.budget_hours})
    persist()
    monitor = None
    try:
        journal.run("hardware", ["nvidia-smi", "-q"])
        journal.run("topology", ["nvidia-smi", "topo", "-m"])
        monitor = journal.launch("gpu-samples", ["nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                                                   "--format=csv", "-l", "1"])
        # Run hash verification in a timed child, so even slow cloud disks cannot
        # consume unbounded billed GPU time. Public data only; no downloads.
        verify_out = folder / f"verify-{time.time_ns()}.json"
        verify_plan = folder / f"selected-plan-{time.time_ns()}.json"
        save_new(verify_plan, {**plan, "models": models})
        journal.run("verify-model-files", [sys.executable, "tools/stage_models.py", "verify", "--plan", str(verify_plan),
                                          "--root", str(Path(args.models_root).resolve()), "--output", str(verify_out)])
        from flow_engine.config import ModelConfig
        for m in models:
            ModelConfig.read(model_dir(args.models_root, m))
        for name, m, dtype, attention in workflow:
            previous = state["stages"].get(name, {})
            if previous.get("status") == "passed" and artifacts_valid(previous):
                journal.event(event="resume_skip", stage=name)
                continue
            attempt = folder / f"{name}-{time.time_ns()}"
            attempt.mkdir()
            state["stages"][name] = {"status": "running", "attempt": str(attempt)}
            persist()
            outputs = []
            if name == "unit":
                journal.run(name, [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
                marker = attempt / "passed.json"
                save_new(marker, {"unit_subprocess_exit": 0})
                outputs = [marker]
            elif name == "cuda-kernel":
                outputs = [attempt / "kernel.json"]
                journal.run(name, [sys.executable, "tools/validate_cuda.py", "--output", str(outputs[0])])
            elif name.endswith("-http"):
                outputs = [attempt / "smoke.json", attempt / "benchmark.json", attempt / "manifest.json"]
                argv = [sys.executable, "-m", "flow_engine.cli", "serve", str(model_dir(args.models_root, m)),
                        "--device", "cuda", "--dtype", dtype, "--attention", attention,
                        "--port", str(args.port), "--max-context", "2048", "--max-active", "4",
                        "--no-prefix-cache", "--log", str(attempt / "engine.jsonl")]
                # Never contact an unrelated pre-existing service on our port.
                import socket
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", args.port))
                proc, thread = journal.launch(name + "-server", argv)
                try:
                    base = f"http://127.0.0.1:{args.port}"
                    wait_ready(proc, base, min(journal.deadline, time.monotonic() + 300))
                    save_new(outputs[2], {"model_digest": m["revision"], "tokenizer_digest": m["revision"],
                                         "dtype": dtype, "hardware": identity["gpu_identity"], "cache_regime": "disabled",
                                         "engine_version": identity["code_sha256"], "launch_command": argv})
                    journal.run(name + "-smoke", [sys.executable, "tools/smoke_http.py", "--base-url", base, "--output", str(outputs[0])])
                    journal.run(name + "-benchmark", [sys.executable, "tools/benchmark_http.py", "run", "--base-url", base + "/v1",
                                                       "--manifest", str(outputs[2]), "--output", str(outputs[1]), "--repeats", "5", "--max-tokens", "128"])
                finally:
                    journal.finish(name + "-server", proc, thread, terminate=True)
            else:
                outputs = [attempt / "correctness.json"]
                journal.run(name, [sys.executable, "tools/validate_model.py", str(model_dir(args.models_root, m)),
                                   "--device", "cuda", "--dtype", dtype, "--attention", attention,
                                   "--tokens", "64", "--draft-tokens", "4", "--output", str(outputs[0])])
            state["stages"][name].update(status="passed", artifacts={str(o): digest_file(o) for o in outputs})
            persist()
            if monitor[0].poll() is not None:
                raise RuntimeError("GPU sampler stopped unexpectedly; resource evidence is incomplete")
        state["status"] = "passed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for entry in state["stages"].values():
            if entry["status"] == "running":
                entry.update(status="failed", error=state["error"])
        raise
    finally:
        if monitor:
            journal.finish("gpu-samples", *monitor, terminate=True)
        state["invocations"][-1].update(elapsed_s=time.time() - started, status=state["status"])
        persist()
        print(f"Campaign {state['status']}: {state_path}. GPU INSTANCE IS STILL BILLED: stop it in your provider console.")


if __name__ == "__main__":
    main()
