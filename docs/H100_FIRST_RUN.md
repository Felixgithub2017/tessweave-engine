# H100 首轮验证与预下载指南

本轮用一张 H100 80GB 验证 Flow 自己的 CUDA 前向、分页注意力、真实权重及 HTTP 服务。目标是找到问题并保留可复现证据，不是宣布支持全部 Arena 模型，也不是跑训练。先验证 Qwen2.5 0.5B，再依次扩大到 1.5B、3B、7B。任何一关失败，停止扩大规模。

## 租多久以及下载是否计入

建议按小时租用，首轮预留 **4 小时 GPU 时间**。这不是已实测的运行时间，也不需要预付整天。若遇到实现错误，保存日志、停止 GPU 计费、回本地修复；只有定位清楚后再启动。8 小时是人工排障预算上限，不是让脚本自动续租的授权。

| 工作 | 首轮工程预算 | 在哪里完成 |
|---|---|---|
| 确定机型库存、云盘挂载、网络与模型下载 | 单独计算 | CPU 实例或厂商提供的存储导入功能 |
| 环境启动、驱动检查、权重哈希检查 | 20～40 分钟 | GPU 实例；软件环境尽量预制 |
| 首次 Triton 编译及分页算子数值检查 | 10～30 分钟 | H100 |
| 四个模型的精度、缓存和服务验收 | 30～90 分钟 | H100，逐个加载 |
| 分析结果、有限重试和打包证据 | 30～60 分钟 | H100 或关闭 GPU 后的 CPU 实例 |

这些区间不是可叠加的精确报价。纯计算可能更短，首次兼容性问题可能使试验提前失败。当前自动化 HTTP 测试是短上下文顺序请求 smoke，不是完整的并发和长上下文性能矩阵。跨框架全面比较另开批次。

下载耗时用实际字节和实测持续速度计算：`秒数 = 待下载字节 / 每秒有效字节`。Mbps 是比特，MB/s 是字节，不能混用。四个模型按名义参数量的 BF16 裸权重约 24GB；以下载清单中的实际文件总大小为准，规划时先按 25～30GB。

以 30GB 十进制传输量为例，持续速度 10/50/100 MB/s 对应约 50/10/5 分钟，另加握手、重试、写盘和哈希时间。1TB 在持续 100 MB/s 下也要约 2.8 小时，不能按小模型的下载经验规划大型 MoE。

CPU 预下载的经济性：`省下的 GPU 下载等待费用 > CPU 下载费用 + 增量存储费用 + 搬运费用` 才划算。小模型且网络很快时，直接在 GPU 上下载可能更省步骤；后续大模型强烈建议预下载。

## 先选 GPU 所在区域再建持久盘

CPU 实例预下载，然后把同一持久盘挂给 GPU，是推荐方案，但先向厂商确认以下事实：

1. 有目标区域的 H100 库存，而且 CPU 和 GPU 实例都能挂载这个存储产品。
2. 它是独立持久盘或网络文件系统，不是删除实例时一起消失的系统盘/容器盘。
3. 普通块存储通常需要卸载并从 CPU 解绑后再挂 GPU；没有集群文件系统时不要同时读写挂载。
4. 同区域不一定等于同可用区；跨区挂载、快照复制和出网可能另收费。
5. GPU 终止之后磁盘保留。保存下载进度所需的 `.cache/huggingface`，不要随意清理。

例如 Runpod 已公布 CPU Pod 网络卷支持，但具体区域库存和挂载限制仍以租用时控制台为准：[CPU 网络卷说明](https://www.runpod.io/blog/enhanced-cpu-pods-docker-network)。不能将这个能力推广为所有云厂商都支持。

首轮建议 CPU 预下载机 2～4 vCPU、8～16GB RAM；GPU 实例至少 64GB 主机内存，优先 128GB、8～16 vCPU、100～200GB 持久盘。不要在内存不足的 CPU 下载机上加载模型。存储若吞吐较低，可在 GPU 启动后复制到本地 NVMe；复制时间同样属于计费时间。

## 下载阶段只需要轻量 Python 环境

以下命令在你核对后手动执行。工作目录是本仓库。示例 `/workspace` 必须确实是保留的挂载点；不是的话替换所有路径。

```bash
conda create -n flow-stage python=3.11 pip -y
conda activate flow-stage
python -m pip install 'huggingface-hub>=0.30,<1' 'requests>=2.32,<3'
mkdir -p /workspace/flow-artifacts
python tools/stage_models.py plan --output /workspace/flow-artifacts/stage-plan.json
```

这一步只读取公开模型 API，固定每个仓库的完整 commit，登记文件大小和发布方内容摘要，不下载权重。默认四个 Qwen2.5 模型；只做第一关可加 `--repo Qwen/Qwen2.5-0.5B-Instruct`。

国内镜像可用 `--source hf-mirror` 生成另一份清单。脚本不自动切源；镜像生成的元数据属于镜像证据，不是独立核实的官方原始签名。首轮清单支持公开 HF 风格 safetensors 仓库，暂不支持魔搭 API 或需要登录的 gated 模型。模型发布要求和许可证仍适用。

预览与下载分开：

```bash
python tools/stage_models.py download \
  --plan /workspace/flow-artifacts/stage-plan.json \
  --root /workspace/flow-models \
  --output /workspace/flow-artifacts/download-01.json

# 审核文件列表、总字节、存储路径和源后，明确批准下载
set -o pipefail
python tools/stage_models.py download \
  --plan /workspace/flow-artifacts/stage-plan.json \
  --root /workspace/flow-models \
  --output /workspace/flow-artifacts/download-01.json \
  --execute 2>&1 | tee /workspace/flow-artifacts/download-01.log
```

预览不会创建报告或下载文件。真正执行后按 `组织--仓库/commit/` 存放权重；不是把最新 main 覆盖到旧目录。脚本关闭 HTTP 环境代理和客户端系统代理发现，不读取 HF 登录令牌，不启用 Xet/HF transfer，不下载或执行模型 Python 文件。操作系统或机房的透明代理不在 Python 控制范围内。

网络中断后重跑同一 plan、同一 root，改用新的 `download-02.json` 和日志名。HF SDK 根据保留的下载元数据继续处理未完成文件。结束后完整校验大小和内容摘要；哈希失败会阻断，不会写“验证成功”。下载只代表文件完整，不代表模型能推理。[HF 下载与固定版本说明](https://huggingface.co/docs/huggingface_hub/en/guides/download)

## GPU 环境准备

优先使用厂商有匹配 NVIDIA 驱动的 Linux 镜像。CPU 下载环境不需要 CUDA，也不能把 Mac 的 conda 环境目录搬到 Linux。

```bash
conda create -n flow-h100 python=3.11 pip -y
conda activate flow-h100
# 示例使用官方 CUDA 12.8 wheel 源。先检查驱动是否兼容。
python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test,cuda,staging]' 'transformers==4.51.3'
nvidia-smi
python -m flow_engine.cli doctor
```

必须看到 `cuda: true`。CUDA wheel 的具体可用版本和驱动兼容性以 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 和厂商镜像说明为准；上面的安装配方尚未在 H100 实测，不是锁定的已验收环境。首次成功后保存解析出的包版本、容器 digest 和驱动版本，再做后续对比。下载 Python/CUDA 包也有网络耗时，尽量提前在同架构 Linux 镜像构建阶段完成。

不要将包含个人凭据的完整 `env` 或 `pip freeze` 直接公开。实验运行器只记录包名和版本，不导出环境变量与私有包安装 URL。

## 从预览到执行

先限制为 0.5B：

```bash
python tools/run_h100.py \
  --plan /workspace/flow-artifacts/stage-plan.json \
  --models-root /workspace/flow-models \
  --output /workspace/flow-artifacts/h100-qwen05 \
  --only Qwen/Qwen2.5-0.5B-Instruct
```

只打印阶段和目录，不安装、不下载、不租机、不调用 GPU。核对后重复命令并添加 `--execute --budget-hours 4`。

每个真实模型执行以下关卡：

1. **文件检查**：重新核验已选模型的全部文件摘要；拒绝缺失/额外权重文件。
2. **单元测试与 CUDA 算子检查**：36组精度、头维度和 GQA/MQA 组合，每组包含不同长度与打乱的物理页。没有 CUDA 直接失败，不能用 skipped 当成功。
3. **FP32 SDPA 参考对照**：排除低精度干扰，比较 logits 和固定长度生成。
4. **BF16 SDPA 参考对照**：确认精度变化后仍满足当前严格门槛。
5. **BF16 Triton 对照**：验证原生分页 kernel 与完整模型的集成。
6. **真实 HTTP 服务**：启动服务，等待权重加载与 warmup 完成，检查 JSON/SSE 一致性，再运行5轮顺序请求 benchmark。前缀缓存明确关闭，避免冷热混测。

遇到 BF16 近似并列 token 的分歧，当前脚本会保守阻断。保留 JSON 对照分析原因，不要放大容差或关掉失败门槛来获得绿色结果。算子验证、模型数值、文本质量、速度是不同证据。

第一关通过后，去掉 `--only`、换一个输出目录，执行四模型完整实验。完整方案先从 0.5B 回归，再向大模型推进。

中断续跑使用相同参数、`--execute --resume`。只有代码、包版本、GPU身份、模型清单及配置没有变化，且通过关卡的结果文件没有修改，才会跳过该关卡。失败关卡保留旧文件，在新 attempt 目录重做。修代码或换 GPU 后必须新开实验目录，不能混用前一次的成功标记。

`--budget-hours` 是每次调用的子进程运行预算，不是云厂商自动停机功能。超时或 Ctrl+C 会停止本轮启动的服务和采样进程，但 **GPU 实例仍然收费，必须在控制台停止/释放**。释放前确认输出在持久盘，并备份证据。

## 如何读实验结果

`state.json` 是逐关状态，`commands.jsonl` 是开始/结束/退出码记录。每个命令的 stdout 与 stderr 合并写到对应 `.log` 并同步终端显示。各 attempt 保存 correctness、smoke、benchmark、manifest 和引擎 JSONL。`gpu-samples.log` 每秒记录GPU利用率、显存、功耗和温度；它属于整张卡，不是某个请求独占资源。

当前 HTTP benchmark 测量首个文本事件延迟和总响应时间，不能用 SSE 分片数代替 token 数，也不能将 completion_tokens/总时间标成纯 decode TPS。短 smoke 的5次重复不能支撑可靠 p95 或“比 vLLM 快”的公开结论。

第二轮才执行同硬件、同权重、同精度、同模板的 vLLM/SGLang 对照，覆盖输入128/2K/8K、并发1/4、冷/热前缀，各条件至少30次；加入 Nsight trace 和逐 token 时间。每次只驻留一个引擎。量化、投机解码和不同输出长度必须单独解释，不能把不等价的结果合并成一个加速比。

## 现在的交付边界

本地测试可以验证下载清单规则、摘要检查、命令日志、超时、续跑条件和现有CPU/MPS数值路径；不能替代真实 H100 的 kernel 编译、驱动兼容和性能结果。预下载脚本提供 CPU 到持久盘的操作流程，不调用云服务商 API 自动创建实例。这个批次完成的是独立引擎的单卡起点，不扩大 Arena20 的支持声明。
