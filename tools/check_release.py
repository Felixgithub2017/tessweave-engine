"""Run local CPU tests, parse source, validate documentation links and archives."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
import tarfile
import unittest
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    errors = []
    sources = [p for folder in ("flow_engine", "tests", "tools") for p in (root / folder).rglob("*.py")]
    for path in sources:
        try:
            tree = ast.parse(path.read_text())
            if path.is_relative_to(root / "flow_engine"):
                for node in ast.walk(tree):
                    imports = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                    if any(n.split(".")[0] in {"vllm", "sglang"} for n in imports):
                        errors.append(f"Upstream runtime dependency: {path}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    links = 0
    for path in [*root.glob("*.md"), *(root / "docs").glob("*.md"), *(root / "validation").glob("*.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if link.startswith(("https://", "http://", "#")):
                continue
            target = link.split("#")[0]
            links += 1
            if not (path.parent / target).exists():
                errors.append(f"Broken local link: {path.name} -> {target}")
    artifacts = []
    for archive in (root / "dist").glob("*"):
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
        elif archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as t:
                names = t.getnames()
        else:
            continue
        forbidden = [n for n in names if n.endswith((".safetensors", ".bin")) or "/runs/" in n or "/.git/" in n]
        if forbidden:
            errors.append(f"Unexpected artifact contents: {forbidden}")
        artifacts.append({"file": archive.name, "files": len(names), "bytes": archive.stat().st_size})
    report = {"tests_run": result.testsRun, "passed": result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
              "skipped": [{"test": str(t), "reason": why} for t, why in result.skipped],
              "test_failures": len(result.failures), "test_errors": len(result.errors),
              "python_sources_parsed": len(sources), "local_document_links_checked": links,
              "release_archives_checked": artifacts, "errors": errors,
              "scope": "Local tests/archive checks. CI and CUDA execution not implied."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not result.wasSuccessful() or errors or not artifacts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
