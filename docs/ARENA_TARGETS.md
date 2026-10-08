# LMArena 模型范围与无权重验证

Flow 的新模型适配范围以 [LMArena Text Overall](https://arena.ai/leaderboard/text) 为准，不再混合 Hugging Face 能力榜、下载量榜或热度榜。当前清单对应页面标注的 **2026年10月2日** 排名，核查日期为 **2026年10月8日**。这是一份适配目标清单，不是“已支持20个模型”的发布声明。

先按榜单顺序筛出非 Proprietary 标签的条目，再映射到厂商公开仓库。同一检查点的思考档位不重复计数；明确不同的检查点保留。榜单的许可证标签不代替实际仓库许可证。API 服务的隐藏提示词、采样配置和线上权重修订未必公开，因此找到同名仓库也不代表能够复现 Arena 分数。

## 二十个适配目标

“总榜名次”包含闭源模型，不是筛选后的开放权重名次。下表顺序才是本次适配顺序；工程实现可以按共享架构分组推进。括号中的架构来自公开 config，不是根据品牌猜测。

| 序号 | 总榜名次 | Arena 名称 | 官方仓库候选与 config 架构 |
|---|---:|---|---|
| 1 | 16 | kimi-k3-max | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) · kimi_k3 / kimi_linear |
| 2 | 26 | mimo-v2.6-pro | [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) · mimo_v2 |
| 3 | 27 | glm-5.3-max | [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) · glm_moe_dsa |
| 4 | 32 | glm-5.2-max | [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) · glm_moe_dsa |
| 5 | 38 | deepseek-v4.1-flash-max | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · deepseek_v41 |
| 6 | 41 | glm-5.3-flash | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · glm5_next |
| 7 | 49 | mimo-v2.5-pro | [MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) · mimo_v2 |
| 8 | 55 | glm-5.1 | [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) · glm_moe_dsa |
| 9 | 56 | deepseek-v4-pro-high-20260813 | [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) · deepseek_v4 |
| 10 | 58 | kimi-k2.6 | [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) · kimi_k25 / kimi_k2 |
| 11 | 62 | deepseek-v4-pro | [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) · deepseek_v4 |
| 12 | 63 | glm-5 | [GLM-5](https://huggingface.co/zai-org/GLM-5) · glm_moe_dsa |
| 13 | 67 | hy3 | [Hy3](https://huggingface.co/tencent/Hy3) · hy_v3 |
| 14 | 76 | gemma-4-31b | [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) · gemma4 / gemma4_text |
| 15 | 77 | mimo-v2.6-flash | [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) · mimo_v2 |
| 16 | 78 | kimi-k2.5-thinking | [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) · kimi_k25 / kimi_k2 |
| 17 | 92 | qwen3.5-397b-a17b | [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) · qwen3_5_moe |
| 18 | 93 | glm-4.7 | [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) · glm4_moe |
| 19 | 95 | inkling | [Inkling](https://huggingface.co/thinkingmachines/Inkling) · inkling_mm_model |
| 20 | 96 | minimax-m3 | [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) · minimax_m3_vl |

第72名 `deepseek-v4-pro-high-preview` 暂归入原版 V4-Pro，不另占一个名额；0813 是明确的另一个检查点，单独保留。此去重是工程范围决策，不是线上端点权重相同的证明。MiMo-V2.6 的 RL 与 MOPD 是不同发布制品，本清单选择 RL 作为候选，不声称它与榜单端点严格等价。

范围首先是**文本输入和文本输出**。一个支持图像的模型进入文本榜，并不意味着它的图像、音频或视频路径也已经通过验证。仍需保留真实多模态包装和文本骨干的语义，不能把嵌套 `text_config` 当成任意 Llama 直接启动。

## 现在能证明什么

[机器可读清单](../flow_engine/data/arena-text-2026-10-02.json)记录模型映射和去重决策；[架构审计报告](../validation/arena-architecture-audit-2026-10-08.json)覆盖20个 config，包含精度、层类型、专家、位置编码、KV 和缺失功能的字段证据。

此次配置来自网页资料的 JSON 重建：4个配置具有固定提交或带固定提交的重定向证据，其余16个来自可变 `main`。报告的 `canonical_config_sha256` 是重建 JSON 的规范化哈希，**不是原文件字节哈希**；单独观察到仓库 SHA 也不能当作 config 已绑定该 SHA。发布可复现的推理配方前，必须重新通过固定提交下载并核对。

本机直接 HTTP 检查20个仓库均超时，见[直连诊断](../validation/arena-metadata-2026-10-08.json)。网页资料可读不代表本机网络可用。工具没有转用付费代理，也没有将网络失败改为成功。

这些模型目前全部未通过 Flow 原生执行的资格关。已有引擎仍只有 README 中列出的稠密 Llama/Qwen2 路径；这里新增的是范围清单、配置审计和资格测试基础设施，**不是20个前向实现**。

## 不下载权重如何开始

在项目根目录执行，下面的审计模块只依赖 Python 标准库，无需先安装 PyTorch：

```bash
# 离线列出冻结的20个候选，不联网、不构造模型。
python -m flow_engine.qualification

# 本机已有config时，离线识别结构并检查原生引擎是否接受。
python -m flow_engine.qualification --config /absolute/path/model/config.json

# 显式联网，仅允许模型API元数据和config.json；不读取本机HF令牌。
# 先获取仓库SHA，再使用固定SHA读取config。默认直连，不读取代理环境变量。
python -m flow_engine.qualification --fetch --workers 4 --timeout 12 \
  --output runs/arena-live-audit.json

# 先检查一个模型，避免网络有问题时重复等待全部模型。
python -m flow_engine.qualification --fetch --repo google/gemma-4-31B-it \
  --output runs/gemma-audit.json

python -B -m unittest discover -s tests -p test_qualification.py -v
```

输出文件已存在时会拒绝覆盖，请使用新文件名。每个元数据响应限制2 MiB，最多4路并发。禁止请求权重、tokenizer 或任意外站重定向，不执行远程 Python。`--fetch` 遇到任何网络失败返回退出码2，并保留其他模型的结果；审计成功的退出码0也不代表模型可推理。

如果由另一台可联网主机采集原始 config，可以通过 `--captured-configs` 在本机离线审计。输入为 `{"models": [{"repo_id": "...", "config": {...}, "config_url": "...", "revision_verified_for_config": false}, ...]}`，必须交代全部20个候选，缺失项用 `error` 明示。该模式不会替你验证外部提供的来源声明。

## 从配置到真正的推理验证

验证按以下次序推进，不能跳过中间的证据：

| 阶段 | 实际执行 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 配置审计 | 解析 JSON，检查字段与实现入口 | 识别结构、定位缺失功能 | 任何矩阵计算正确性 |
| 全尺寸 meta 检查 | 使用真实结构构建无存储张量并检查算子形状 | 维度、权重键、布局衔接 | 数值、GPU kernel、性能、显存峰值 |
| 缩小尺寸数值对照 | 同一架构实现，固定随机参数，对照独立参考 | RoPE、路由、缓存、分块和解码数学语义 | 完整检查点质量及大尺寸 kernel 行为 |
| 真实权重验证 | 加载固定版本模型，校验 logits 和 token 序列 | 特定模型、精度和后端上的推理正确性 | 其他精度/硬件自动正确 |
| GPU性能验证 | 固定工作负载，测量冷/热缓存、TTFT、逐token延迟和峰值显存 | 该配置下的性能与回归 | 跨硬件或任意请求均更快 |

目前20个新候选只完成第一阶段。禁止用全零权重返回固定文本来冒充“模拟推理通过”；禁止用 Transformers/vLLM/SGLang 代跑后声称是 Flow 原生支持。它们可以作为对照组，但被测路径必须执行我们的前向、缓存和调度代码。

## 为什么必须按架构族实现

以一次解码为起点：新 token 先变成隐藏向量，接着经过注意力、前馈和残差，再输出词表 logits。不同架构改变的不是名字，而是这个过程中的状态与计算。

1. **先检查注意力保存了什么。** 普通 GQA 保存 K/V；潜变量注意力可能保存压缩表示；线性注意力维护递归状态。三者不能共享一条未经证明的 KV 字节公式。混合模型还需要按层路由到不同状态管理器。
2. **再检查一个 token 去了哪些专家。** MoE 除了 top-k，还有打分函数、分组限制、共享专家、归一化与缩放。减少专家数量做测试时，必须保留这些分支；用一个普通 MLP 替代所有专家无法验证路由。
3. **检查加载的数据如何还原。** 配置的 BF16 常指计算精度，不代表所有权重均以BF16存储。注意力FP8、专家FP4、缩放因子和排除层需要独立解释。直接把压缩整数当浮点权重会“能运行但答案错误”。
4. **检查投机解码能否回滚全部状态。** 不只是删除输出 token。线性注意力状态、压缩缓存、专家相关辅助状态均需恢复到最后一个确认位置；仅回退普通 KV 长度对混合模型不够。

具体例子：Gemma4配置包含局部/全局层的不同头维度与位置编码；Kimi-K3配置包含线性注意力与潜变量注意力字段；DeepSeek-V4配置包含压缩比例、专家精度和多流残差字段。上述事实的配置来源逐项保存在审计报告中。它们要求不同的实现与测试，不能仅在模型名称白名单里增加一个字符串。

适配顺序建议先完成一个注意力状态与路由族的数值闭环，再扩展同族模型，最后增加量化和分布式执行。用户租GPU前应看到每个目标的缺失功能和明确验收命令；只有原生数值路径完成后，才进入真实权重云端验证。
