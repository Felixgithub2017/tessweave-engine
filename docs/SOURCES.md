# Design references and source boundaries

Reviewed 2026-10-08. Flow's implementation is independent; these links explain established ideas and reviewed upstream behavior. Upstream source checkouts live in the separate inference-source-lab study folder and are not bundled with this package.

## vLLM

Source release v0.31.0, commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`.

- [Scheduler](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py): token-budget scheduling, cache allocation and unfinished-request state.
- [KV cache manager](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_manager.py): physical allocation and prefix reuse.
- [GPU runner](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu_model_runner.py): execution and proposal/verification orchestration.

Flow does not inherit vLLM's kernel coverage, distribution support, graph replay or production validation by using paging terminology.

## SGLang

Source release v0.5.21, commit `e00930c5489053f26d86b179cee0d087f846acbb`.

- [Scheduler](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/managers/scheduler.py): request scheduling and normal/overlap execution paths.
- [Radix cache](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/radix_cache.py): shared-prefix ownership, matching and eviction.
- [Model runner](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/model_executor/model_runner.py): model execution dispatch.

Flow's parent-hash block index is not SGLang's compressed radix tree. Flow currently has no equivalent to its hierarchical cache, speculative algorithm portfolio or execution overlap.

## Numerical implementations

- [PyTorch SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): attention semantics and backend-dependent numerical behavior. Flow uses an explicit offset-aware mask for cached attention.
- [Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html): online-softmax tiled attention background. Flow's paged decode kernel is separately written and uses its own page layout.
- [Transformers Qwen2 v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py) and [Llama v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama/modeling_llama.py): reference implementations for independent forward/token tests.
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775): locally available real checkpoint used in recorded validation; weights are not redistributed here.

Dependency and checkpoint licenses apply separately. No upstream benchmark figure is presented as Flow performance.
