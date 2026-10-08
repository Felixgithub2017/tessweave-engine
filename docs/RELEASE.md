# Publishing the research pre release

Version `0.1.0a1` is a research pre-release. The local repository contains original MIT-licensed engine code, source attribution, tests, user documentation, CI configuration and recorded validation. It does not contain weights or upstream Git checkouts.

## Publication checklist

1. Choose the owner's GitHub account/organization and confirm repository/package-name availability. Flow Inference is currently a working name.
2. Review LICENSE, NOTICE and the support matrix. Configure private security reporting and repository maintainers.
3. Run CPU tests and inspect the skipped CUDA gate. Do not change the badge/documentation to GPU-validated without running it.
4. Run at least the recorded real Qwen FP32 gate. Keep the known FP16 divergence visible in release notes.
5. Build and inspect the sdist and wheel. Confirm no weights, credentials, local run logs or private prompts are included.
6. Publish a Git repository to the explicitly selected destination; create a `v0.1.0a1` pre-release with the support boundary and test evidence.
7. A PyPI release is optional and requires a separate account/name decision; do not infer permission to publish under a guessed identity.

## Local build

```bash
python -m pip install -e '.[test]'
python -B -m unittest discover -s tests -v
python -m build
```

The default setuptools sdist includes project/package/test metadata; the repository is the full teaching/evidence deliverable. Inspect archive contents before publishing. CI definitions are supplied but are not marked as executed until they actually run on the hosting service.

## Suggested release description

Flow Inference 0.1.0a1 is an independent single-device text inference engine for low-concurrency experimentation. It implements physical KV paging, shared full-prefix blocks, decode-first chunked scheduling, packed linear projections and verified greedy prompt lookup. Real Qwen2.5-0.5B-Instruct FP32 correctness was checked on CPU and Apple MPS. CUDA performance and broad architecture support remain open gates. MPS FP16 can diverge from the strict reference. This release does not claim superior speed to vLLM/SGLang or production readiness.

## Public hosting status

The public repository is [Felixgithub2017/flow-inference](https://github.com/Felixgithub2017/flow-inference). The first verified remote commit is `bb7c676925c5581a5d36dad9190295ae851b3c71` (2026-10-08). Its display brand is now TessWeave Engine; repository, package and CLI names remain compatible. Source publication is not GPU validation. No tagged release or PyPI publication is claimed by this milestone.
