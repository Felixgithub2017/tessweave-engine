# 从一条请求推演 Flow Inference 引擎

这里讨论的是我们自己执行模型的引擎，不是包装 vLLM 的控制台。入口是本地 safetensors，出口是模型生成的文本；中间的页表、调度和计算都在本仓库。把它作为第七部分的研究成果，首先要让每个状态变化可验证，再谈性能竞争。

## 先分清楚借鉴和实现

vLLM 让我们重视物理 KV 页、逐轮调度和 token 预算；SGLang 让我们重视共享前缀的生命周期、请求状态与执行张量的分离。Flow 用自己的代码实现这些基础机制，选择较小的支持范围：单设备、文本稠密模型、少量同时请求。

我们没有把 SGLang 的 Radix Tree、vLLM 的 PagedAttention 内核拷贝进来。`cache.py` 用“父哈希＋当前完整 token 块”的链式索引共享前缀；`model.py` 的默认路径从页池取出 K/V 后调用 SDPA；CUDA 可选路径才通过页表直接读缓存。三者不能混称为“已经拥有成熟框架的所有能力”。

```text
POST /v1/chat/completions
  → 校验选项 → tokenizer.chat_template → Engine.submit
  → waiting → 最坏容量准入 → active
  → 先批量 decode，再推进一个 prompt chunk
  → model.forward → KV 写入/读取 → attention → MLP → logits
  → greedy/sample 或 draft verify → 提交经过确认的 token
  → SSE → 完成/取消 → 归还请求引用 → 保留可复用完整前缀
```

主服务线程只处理协议；`flow-model-worker` 是唯一模型执行线程。`Engine` 的锁序列化提交、取消和执行，防止一个请求在计算中途被释放 KV。HTTP 用线程转交来避免等待锁时阻塞整个异步事件循环。这种保守结构容易审计，但不是 GPU/CPU 流水重叠，后续需要通过 trace 量化其损失。

## 第一步把模型变成真的矩阵

从 [ModelConfig.from_dict](../flow_engine/config.py) 开始。`model_type=qwen2` 并不足够，还要检查 full attention、默认 RoPE、SwiGLU、头数整除关系、是否量化、是否要求 remote code。不同 RoPE 会改变位置向量；用错误的位置映射跑出来的文本看起来正常也不能算支持。

[CausalLM.load](../flow_engine/model.py) 的顺序是：

1. 读取配置并拒绝未实现语义。
2. 在 meta device 上建网络，先设目标精度，再在执行设备分配权重；避免先造一份随机权重或在低精度部署前分配完整 FP32 副本。
3. 按 safetensors 索引逐分片加载，不使用 pickle，不运行模型仓库 Python。
4. 对每个参数核对名称、形状、重复和缺失；绑定词嵌入的模型共享 embedding 与 lm_head。
5. 合并 Q/K/V 和 gate/up 的行，释放旧矩阵引用，进入 eval 模式。

若 `Wq∈R^(Hq·D × H)`，`Wk,Wv∈R^(Hkv·D × H)`，则：

```text
Wqkv = concat_rows(Wq, Wk, Wv)
QKV = X @ Wqkv.T
Q, K, V = split_last_dimension(QKV)
```

这是相同线性映射的合并，不是改变模型参数知识。把原本三个独立线性层变成一个较大的调用，减少 launch 和重复读 X 的机会。gate/up 同理。每层从 7 次线性调用降为 4 次，但 GEMM 算量基本没变，整机速度不会自动增加 43%。低精度的累加顺序仍可能改变近似并列的 argmax，必须实测。

## 第二步分配 KV 物理页

[KVPool](../flow_engine/cache.py) 的张量形状是：

```text
K 和 V 各为 [L, physical_blocks, block_size, Hkv, D]
request.table[logical_block] = physical_block
position 37，block_size=16 → logical_block=2，block_offset=5
若 table[2]=7，物理 token slot = 7×16+5 = 117
```

不要把查询头数 Hq 填进缓存大小公式。GQA 中 K/V 只有 Hkv 组头；计算阶段多个查询头共用它们。

以本地验证的 Qwen2.5-0.5B-Instruct 为例：L=24，Hq=14，Hkv=2，D=64。

| 项目 | FP32 | FP16/BF16 |
|---|---:|---:|
| 每个 token 的 K+V 元素 | `2×24×2×64 = 6144` | 同左 |
| 每个元素字节 | 4 | 2 |
| 每 token KV | 24576 B = 24 KiB | 12288 B = 12 KiB |
| 64块×16token 页池 | 24 MiB | 12 MiB |
| 默认256块×16token 页池 | 96 MiB | 48 MiB |

页池是**实际预分配张量**，不是请求用了多少才向驱动申请多少。缓存复用节省的是池内占用和重复计算，默认不会让驱动统计的整块池内存下降。

还要另算权重、分片加载临时张量、合并投影时的单层副本、激活、SDPA gather、GQA repeat、attention mask、运行时和分配器。`inspect` 只给 KV 的精确张量账，不把这一个数说成“总显存”。

## 第三步共享前缀但不能共享可变尾巴

假设每块4个 token，A 的输入是 `[1,2,3,4,5,6]`。只有 `[1,2,3,4]` 是可公开共享的完整块；尾部 `[5,6]` 仍属于请求私有空间。

每个 key 由 namespace、父块哈希和当前4个 token 决定。因此 `[9,9,9,9,1,2,3,4]` 的第二块不会与以 `[1,2,3,4]` 开头的请求误共享：位置与历史不同，K/V 也不同。

`publish` 只发布已经**计算并确认**的完整块。`acquire_prefix` 最多命中到 prompt 倒数第二个 token，以保留至少一次前向来得到最后位置的 logits。它不假定“所有 K/V 命中就已经有下一个 token 的分布”。

每个块有两类引用：请求引用和缓存索引引用。请求完成只还请求引用；LRU 淘汰只还缓存引用；两者都归零才进入 free list。共享块永不写入，新增 token 写进私有尾部。当前版本在请求正常结束时发布前缀，不在正在计算的请求之间提前公开未完成状态。

## 第四步准入和调度

`Engine._admit` 先获得可用完整前缀，再给请求预留：

```text
总需要块数 = ceil((prompt_length + max_tokens) / block_size)
新增块数 = 总需要块数 - 已复用块数
```

这种策略牺牲了一部分过量承诺能力，但一个被接纳的请求不会因为自己按声明正常生成而突然无页可用。容量不足时先淘汰缓存引用，然后 FIFO 等待；不会杀掉其他正在生成的请求来掩盖容量不足。大请求可能挡住后面的短请求，这是明确的 head-of-line blocking 改进点。

每轮 `Engine.step`：

1. 接纳新请求。
2. 找出所有 prompt 已完成的请求，每个推进一个 decode token，一次批执行。
3. 在未完成 prompt 之间轮转，推进一个 chunk。
4. 测量真实 step 耗时，调整下一次 chunk 的上限。

这里不是“每个请求从头到尾独占 GPU”。但也不是把所有不同长度的 prefill 合成高效 ragged kernel：第一版每次只推进一个 prefill，先把状态正确性和解码等待控制住。

反馈近似为：

```text
prefill_ms_per_token = EMA(本次prefill耗时 / 本次token数)
剩余预算 = max(1ms, target_step_ms - decode_ms)
下一chunk = clamp(剩余预算 / EMA, min_chunk, max_chunk)
```

没有硬件配置能保证“50ms一定完成”；长上下文算量、系统负载、CPU采样、内核编译都会改变实际时间。即使预算不足也推进最小 chunk，避免长 prompt 饥饿。EMA 是启发式，不是最优控制证明。

## 第五步跟一层 Transformer

[CausalLM.forward](../flow_engine/model.py) 输入 `ids[B,T]` 和绝对 `positions[B,T]`。批解码的 T=1；当前 prefill 的 B=1。

```text
embedding
→ RMSNorm（方差用FP32算）
→ packed QKV projection
→ 给Q/K施加RoPE
→ 把K/V按slot写入页池
→ 取历史K/V做attention
→ O projection + residual
→ RMSNorm
→ packed gate/up → SiLU(gate)×up → down
→ residual
→ 末层norm和lm_head → logits
```

重要陷阱：已有100个历史 token，当前 query 只有1个时，不能直接把 `is_causal=True` 当成正确的偏移 mask。当前 query 的绝对位置是100，应看到0到100，而不是只看到 key0。参考路径显式构造 `key_position <= query_absolute_position`，并屏蔽批内 padding。测试专门比较分块/整段与不同长度批请求。

默认 SDPA 路径为了可移植会 gather 全历史 K/V，并显式扩展 GQA 头。它是可读、可验证的起点，也是已知带宽浪费，不能包装成“比上游更高效的 PagedAttention”。

[Triton decode](../flow_engine/kernels/paged_decode.py) 则让每个 `(request, query_head)` program 沿页表直接取真实 K/V，按 GQA 映射找 kv_head，并分 tile 做在线 softmax：

```text
m_new = max(m_old, max(scores_tile))
alpha = exp(m_old - m_new)
acc_new = acc_old×alpha + sum(exp(scores_tile-m_new)×V_tile)
z_new = z_old×alpha + sum(exp(scores_tile-m_new))
最终 attention = acc / z
```

这样不需要构造整段 score 向量，也不需要重复展开 K/V。但是单头单 program 在长上下文可能并行度不足；页表当前仍由 CPU 逐轮构建上传；没做 split-K、图捕获或持久化缓冲区。这是等待 CUDA 实验的具体优化空间。

## 第六步自己验证投机续写

只给 `--draft-tokens 1..8` 且只有一个活跃贪心请求时启用。`draft.propose` 在已经提交的历史中找最近的3到6个 token 后缀，取其历史后续作为建议。建议本身没有任何输出权。

假设下一次输入本来是 `x`，草稿提出 `[a,b,c]`。我们一次执行 `[x,a,b,c]`，模型给出四组 logits。依次核对：

```text
argmax(logits[0]) 是否等于 a？
argmax(logits[1]) 是否等于 b？
argmax(logits[2]) 是否等于 c？
```

若第一项相同、第二项不同，只提交 `a` 和模型在第二项的真正 argmax。后面的位置即使已写入物理 KV，也不能进入逻辑长度或共享前缀。下次计算会覆盖它们。全部一致则还能提交最后一组 logits 的 bonus token。

`computed` 表示已确认可用 KV 的长度，而不是“内核算过多少位置”。这一个区分决定了投机拒绝是否会污染后续 attention。测试覆盖全部接受、全部拒绝、后续前缀复用、采样禁用以及 EOS 边界。

这里没有随机采样的接受—拒绝校正；温度大于0会直接退回常规解码。低接受率、长验证成本都可能让它更慢，默认关闭。下一阶段应基于真实接受率与验证成本控制 K，而不是凭“投机”二字就默认开启。

## 第七步观察而不伪造

[Trace](../flow_engine/trace.py) 记录提交、页预留、命中、prefill/decode/verify、首次 token、完成和失败。`/metrics` 提供页池和请求统计。它们不是 GPU 占用率或逐算子 profiler 数据；没有接探针就不显示虚构的 GPU 矩阵动画。

`first_token` 是引擎生成第一个 token 的时间；HTTP benchmark 的 `first_text_s` 是收到第一段可显示文本的时间。Unicode/整词缓冲会让两个时间不同。SSE事件数量不是生成 token 数，只有服务端 usage 才用于计算生成数量。

当前回放是 JSONL 状态轨迹回放，不记录私人 prompt 或完整回答。需要做性能实验时，benchmark 报告显式保存公开测试用例的回答，用于不同引擎的质量一致性检查。

## 哪些问题还没有解决

MoE、量化、MLA、QK Norm、扩展 RoPE、视觉编码器、多设备通信、CUDA Graph、异步重叠、KV量化、长上下文分段归约和推测式随机校正都不是第一版功能。引擎显式拒绝不支持的模型；路线见 [ROADMAP](ROADMAP.md)。

目前的成果是一套真正执行真实权重、能解释且能进行源码级消融的独立引擎。要成为工业级竞争产品，还必须经过更广泛架构验证、真实 CUDA 性能、可靠性、负载与安全测试；这些门禁不能由教程写得足够长来替代。
