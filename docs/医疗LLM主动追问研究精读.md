# 医疗 LLM「主动追问 / 信息寻求」四篇核心论文精读笔记

> 主题：围绕"该追问时不追问（context-seeking / information-seeking failure）"这一失败模式，
> 精读四篇论文的实验设计与结论，并做批判性评估。
> 精读日期：2026-11-24

---

## 0. 四篇论文总览

| 论文 | 出处 / 年份 | 是否真交互式 | 核心贡献 | 一句话结论 |
|---|---|---|---|---|
| **BODHI** | BMJ Health & Care Informatics, 2026（MIT 主导） | ❌ **不是** | Prompt 工程让模型"表现出谦逊/追问" | 只是让回复里**带上追问语句**，无信息获取闭环 |
| **MediQ** | NeurIPS 2024（Stanford 等） | ✅ 是 | 双 agent 交互 + Abstention 置信度机制 | 裸追问有害；加置信度弃权才有效，但仍差上界 10.3% |
| **Hx2Dx ARENA** | arXiv:2611.14122, 2026 | ✅ 是 | NEJM 1000 病例的交互式诊断竞技场 | 所有模型交互式掉 25–36pp；瓶颈在"问"不在"推理" |
| **OncoRounds（信息寻求失败）** | arXiv:2607.10275, 2026 | ✅ 是 | 血液肿瘤三轮 agentic 框架，32 模型 | 最好仅 68%；锚定/过早闭合/满意即止仍是主因 |

---

## 1. BODHI —— 重点精读

### 1.1 论文信息

- **全称**：*An Engineering Framework for Curiosity-Driven and Humble AI in Clinical Decision Support*
- **出处**：BMJ Health & Care Informatics, 33(1): e101877（DOI: 10.1136/bmjhci-2025-101877）
- **团队**：MIT Critical Data（Leo Anthony Celi 为资深作者）主导，联合 Sorbonne、Melbourne、Harvard-MIT、ETH Zurich、UCL 等
- **配套哲学版**：*Beyond Overconfidence: Embedding Curiosity and Humility for Ethical Medical AI*（PLOS Digital Health, 2026, DOI: 10.1371/journal.pdig.0001013）—— 这篇是设计理念，无实验数字

### 1.2 机制：它是怎么做的

BODHI 是 **two-pass（两段式）prompt 工程**，**不是新模型、不是微调、不是 agent、更不是多轮交互系统**：

1. **Pass 1（内部不确定性分析）**：强制模型先输出一段结构化分析，覆盖 7 个维度：
   - 任务类型分类（Emergency / Technical / Hybrid / Conversation）
   - 受众识别（患者 / 医护 / 不明确）
   - 主假设 + 推理
   - 影响置信度的关键不确定性
   - **澄清问题（非急诊场景强制要求 1–2 个）**
   - 触发转诊的红旗（red flags）
   - 与该不确定性水平匹配的安全建议
2. **Virtue Activation Matrix（美德激活矩阵）**：把「置信度 × 临床复杂度」映射到 4 个 stance：
   - Q1 高置信/低复杂 → Proceed & Monitor（直接建议 + 监测点）
   - Q2 高置信/高复杂 → Watchful & Alternatives（主方案 + 备选 + 升级条件）
   - Q3 低置信/低复杂 → Clarify & Review（先问 1–2 个关键问题）
   - Q4 低置信/高复杂 → Escalate & Reframe（承认局限、转诊、给安全步骤 + 红旗）
3. **Pass 2（对外回复）**：根据 stance 生成最终给临床医生看的回复。

### 1.3 实验设置（关键，逐项拆解）

| 项目 | 实际设置 |
|---|---|
| 数据集 | **HealthBench Hard 的 200 个临床场景（vignettes）**——注意不是全量 1000，是抽的 200 |
| 待测模型 | **GPT-4o-mini 和 GPT-4.1-mini**——两个都是 **mini 小模型**，不是旗舰模型 |
| 种子 | 5 个随机种子，每模型 200×5=1000 条观察，共 2000 条 |
| 干预 | 纯 prompt 层（两段式 CoT），**无任何训练或架构改动** |
| 评分 | rubric 打分，主要由模型评分器完成（论文自认缺 clinician-in-the-loop 验证） |

### 1.4 关键数字

| 指标 | GPT-4.1-mini（基线→BODHI） | GPT-4o-mini（基线→BODHI） |
|---|---|---|
| Context-seeking（追问率） | 7.8% → **97.3%**（d=16.38） | 0% → **73.5%**（d=19.54） |
| Hedging（保留/对冲） | 1.7% → 21.9%（d=5.80） | — |
| 临床质量综合分 | 2.5% → **19.1%**（+16.6pp） | 0% → 2.2%（+2.2pp） |
| **沟通质量** | **↓ 12.5pp（d=−2.94）** | **↓ 11.3pp** |

### 1.5 批判：它真的是"不断问问题补全信息最后给诊断"吗？——**不是**

这是本笔记最重要的结论，直接回答"它的实验到底是啥"：

1. **BODHI 不是交互式问诊。** 它跑在 **HealthBench Hard** 上，而 HealthBench 是**静态**任务——给模型一段已发生的对话/场景描述，让它**生成一条回复**。"追问"在这条回复里体现为**一句澄清问题**（例如"请问您最近有没有发烧？"），仅此而已。

2. **没有信息获取闭环。** 模型问出问题之后，**并不会真的得到答案、再据此继续推理、最终收敛到诊断**。它是在**同一条静态回复里**，既问了问题，又（带着条件地）给了建议/诊断。这与 MediQ、Hx2Dx、OncoRounds 里"问→拿到答案→再问→…→才诊断"的多轮闭环**完全是两回事**。

3. **"97.3% 追问率"是近循环（near-circularity）指标。** 独立评论（Aiomics）一针见血：*prompt 明确指示系统"要问澄清问题"，而头条指标就是"数澄清问题"*。这么大的效应量（d=16.38）证明的是**模型服从了指令**，而不是**它在正确的时刻追问、或追问改变了诊断**。

4. **基线低得离谱，放大效应。** 基线模型是 GPT-4.1-mini（质量分 2.5%）和 GPT-4o-mini（0%）。从 0 涨到 19%，看起来惊人，但 **19.1% 的绝对水平仍然远未可用**，且不能外推到 GPT-5 级旗舰模型（它们本身的追问率没这么惨、提升空间也没这么大）。

5. **有代价没被重视：沟通质量下降 12pp。** 论文把下降解释为"基准在惩罚恰当的保留"，有道理，但这是真实 trade-off——同一套干预把 curiosity 拉高、把 communication 压低。

6. **与 Hx2Dx 的结论正面冲突。** Hx2Dx 的过程指标证明：**诊断失败不是"问得少"，而是"问得没鉴别价值"**。BODHI 恰恰只优化了"问没问"，没优化"问得对不对"。**追问率 97% 与"问得对"是两码事。**

> **一句话批判结论**：BODHI 证明的是"用 prompt 能让模型按要求表现出谦逊/追问"，它**没有**证明谦逊是"校准好的"（在正确时机才保留），**没有**证明追问是"有鉴别价值的"，**也完全没有**任何多轮信息获取→再诊断的闭环。把它当成"解决了 context-seeking"是对论文边界的误读。

---

## 2. MediQ —— 精读

### 2.1 论文信息

- **全称**：*MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning*
- **出处**：NeurIPS 2024（Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Emma Pierson, Pang Wei W. Koh, Yulia Tsvetkov）
- **定位**：真正把"静态 QA"改成"交互式"的开创工作

### 2.2 机制（真正的交互式）

- **Patient System（患者代理）**：持有隐藏的完整病历；用 **Fact-Select** 把病历拆成原子事实，只回答与当前问题相关的事实，避免泄露/幻觉。
- **Expert System（被评模型）**：初始只拿到年龄/性别/主诉，每轮二选一：**追问** 或 **提交最终诊断**（多选题）。
- **Abstention Module（弃权模块）**：三种策略组合，先判断"该不该继续问"：
  - **Scale**：对置信度打 1–5 分
  - **Rationale Generation**：强制解释"为什么自信/不自信"
  - **Self-Consistency**：多次采样求共识

### 2.3 关键数字

| 指标 | 数值 |
|---|---|
| 最佳 Abstention 组合 vs 基础交互式 | **+22.3%** 准确率 |
| 最佳 Abstention vs "不交互硬猜" | +12.1% |
| 与"全信息上界"的差距 | 仍有 **10.3%**（多轮整合未解决） |
| Patient System 事实性（Fact-Select vs 直接 prompt） | 89.1% vs 55.9% |

### 2.4 批判要点

- **最反直觉、也最重要的发现**：直接 prompt 模型"信息不够就追问"**反而降性能**——模型会问无关/重复问题、多轮整合能力差。**这直接反驳了 BODHI 式"只要让它追问就行"的路线。**
- 局限：最终诊断仍是**多选题**（不是开放诊断）；Patient System 依赖 GPT-4 闭源 API（可复现性差）；数据集缺乏足够细的病历支撑模拟。

---

## 3. Hx2Dx ARENA —— 精读

### 3.1 论文信息

- **全称**：*Hx2Dx: From Anamnesis to Diagnosis — A Multimodal Arena for Conversational Clinical Reasoning in LLMs*
- **出处**：arXiv:2611.14122（REMEDI Lab, Università della Svizzera italiana 等）

### 3.2 机制（真交互 + 多模态）

- **数据**：NEJM 1000 例真实病例（1992–2026），排除无法言语交流者。
- **三组件**：参考诊断 + 模拟患者脚本（只含患者能陈述的信息）+ 调查注册表（每项检查的原始模态结果，含影像 PNG 原图）。
- **动作空间**：`ASK(q)` 提问 / `TEST(r)` 请求检查 / `FINAL_DIAGNOSIS`，上限 50 轮。
- **模拟患者**：OSCE 标准化患者模式，**只答被问的、不主动给**；Data agent **只检索不生成**，防幻觉污染。
- **评分**：Correct / Partially Correct / Incorrect 三档。

### 3.3 关键数字

**静态 vs 交互式诊断准确率（Correct + Partial）：**

| 模型 | 静态 | 交互式 | 损失 |
|---|---|---|---|
| GPT-6 Luna | 83.9% | 47.4% | **−36.5pp** |
| GLM 5.3 Flash | 79.7% | **54.4%** | −25.3pp |
| DeepSeek V4.1 Flash | 75.1% | 50.3% | −24.8pp |
| Gemini 3.5 Flash Lite | 74.7% | 38.5% | −36.2pp |
| Llama 4 Scout | 57.9% | 23.6% | −34.3pp |
| Gemma 3 27B IT | 55.1% | 25.7% | −29.4pp |

**核心实验（transcript-transfer，把"问"和"推理"拆开）：** 把最强访谈者 GLM 5.3 的问诊记录喂给所有模型，让它们只做诊断：

| 模型 | 自己访谈 | 用 GLM 记录 | 提升 |
|---|---|---|---|
| Llama 4 Scout | 23.6% | 48.0% | **+24.4pp** |
| Gemma 3 27B | 25.7% | 44.9% | +19.2pp |
| GPT-6 Luna | 47.4% | 56.3% | +8.9pp |

模型间差距从 32.5pp 缩窄到 11.4pp → **证明短板在"证据获取"，不在"诊断推理"**。

### 3.4 批判要点

- 硬证据说明**所有模型交互式都大幅掉分**，这是"context-seeking 远未解决"的最强佐证。
- 过程指标揭示：**失败不是问得少，而是问得没鉴别价值**（错误病例反而用了更多回合/检查）。
- 自认局限：仅 7 模型；NEJM 病例偏向复杂/不典型；**数据可能已进训练集**（虽交互式部分缓解）；部分专科样本小（产科 17 例等）。

---

## 4. OncoRounds（信息寻求失败，arXiv:2607.10275）—— 精读

### 4.1 论文信息

- **全称**：*Information-seeking failures of large language models in agentic clinical reasoning*
- **出处**：arXiv:2607.10275（2026-07-11，Keno K. Bressem 等）

### 4.2 机制（三轮 agentic，血液肿瘤）

- **数据**：20 个复杂血液肿瘤病例（AML、骨髓瘤、DLBCL、ALL、MDS 等，均为复发/难治/移植后），由 10 年经验血液科医生 **de novo 编写**（防泄漏），7 位专家审核。
- **三轮**：R1 门诊初诊 → R2 住院检查 → R3 确定性表征（释放分子/细胞遗传/治疗史）。
- **动作**：每轮每 turn 只能**请求一条信息**或**提交诊断+治疗方案**（single-request-per-turn，强制优先级判断）。
- **评分**：5 个 GPT-5 Mini 实例组成的评审团 + 多数投票；与 10 年经验血液科专家对照 κ=0.804。

### 4.3 关键数字

| 指标 | 数值 |
|---|---|
| 32 个前沿模型里最好（Claude Opus 4.6） | 仅 **68.1%** 总体准确率 |
| 所有模型平均 | 56.3% |
| 信息利用率 R1/R2 → R3 | ~57% → **25.7%** |
| 信息利用率 vs 准确率相关性 | **R=0.69**（最强预测因子） |
| 鉴别诊断（所有模型最弱项） | 43.6%（比诊断低 23pp） |
| 恢复率（R1 错→R3 对） | 仅 34.4% |
| 恶化率（R1 对→R3 错） | 50.8% |
| 25/32 模型 R3 得分低于 R1 | ✅ |

**三个失败模式（按 Croskerry 双过程模型分类）：**

| 失败模式 | 定义 |
|---|---|
| **Search satisficing**（满意即止） | 找到一个看似合理的解释就停止搜索 |
| **Anchoring**（锚定） | 固守初始印象、无视矛盾信息 |
| **Premature closure**（过早闭合） | 充分验证前就接受诊断 |

错误总检出率 8.8%（1008 次评估中 89 次），anchoring 是最频繁的认知偏差。

**最扎心的对照：** reasoning traces 有 **91.1% 达到"高质量"阈值**，但平均准确率只有 **56.3%**，R-IDEA 与准确率仅 R=0.56 → **"局部自洽的推理链" ≠ "全局正确的结论"**。

### 4.4 批判要点

- 关键证据：**分子/细胞遗传数据在 R3 被系统性遗漏**（利用率骤降到 25.7%），而血液肿瘤的治疗选择恰恰取决于这些。论文归因于"lost in the middle"长上下文检索缺陷。
- 锚定不对称：**初始错误持续存在的概率（65.6%）约为"初始正确被维持"的两倍**——说明模型一旦一开始判错，很难自己纠回来。
- 局限：仅 20 例（样本小）；单领域（血液肿瘤）；合成病例，非真实患者。

---

## 5. 横向对比与批判性总结

### 5.1 四篇的"追问"根本不是一回事

| 论文 | "追问"的真实含义 | 有没有信息闭环 |
|---|---|---|
| BODHI | 在**一条静态回复里**写一句澄清问题 | ❌ 无 |
| MediQ | 多轮交互中决定"问 or 答" | ✅ 有（但最终是选择题） |
| Hx2Dx | 多轮 ASK/TEST 直到 FINAL_DIAGNOSIS | ✅ 有 |
| OncoRounds | 三轮逐步请求数据再诊断 | ✅ 有 |

**因此，把 BODHI 的"97.3% 追问率"和 MediQ/Hx2Dx/OncoRounds 的"交互式诊断准确率"放在一起比较，是拿两个不同层面的指标硬比——这是该领域最容易被带偏的地方。**

### 5.2 三个可迁移的批判性判断

1. **追问率 ≠ 追问质量。** BODHI 把追问率刷到 97%，Hx2Dx 却证明"问得多≠问得对"，OncoRounds 证明"问到最后反而停止问关键数据"。**只看追问率会误判。**

2. **"表现出谦逊" ≠ "在正确的时刻谦逊"。** 校准（calibration）才是关键——该问的时候问、该果断的时候果断。目前没有任何一篇证明模型做到了校准，多数只是"让模型更爱问/更爱保留"。

3. **当前整体状态 = 缓解，未解决。** 最硬的三组证据：Hx2Dx 所有模型交互式掉 25–36pp；OncoRounds 最好只有 68%、锚定恢复率仅 34%；MediQ 与全信息上界仍差 10.3%。**交互式信息获取是当前医学 LLM 真正的天花板，而不是知识量。**

### 5.3 如果要跟人讨论，记住这几句话

- "HealthBench 是静态的，BODHI 在它上面测，所以 BODHI 的'追问'只是回复里的一句话，没有闭环。"
- "MediQ 早就证明裸追问会降性能，BODHI 用 prompt 强制造句追问，恰好踩在这条警告上。"
- "Hx2Dx 的 transcript-transfer 是金标准：把好访谈者的记录给别人，弱的也能追平，说明短板在'问'不在'推'。"
- "OncoRounds 最吓人的是：模型推理链 91% 看着都对，但结论 56% 是错的，而且一开始错就很少能翻回来。"

---

## 附：参考文献

1. Arslan J, Benke K, Cajas Ordóñez SA, et al. *An Engineering Framework for Curiosity-Driven and Humble AI in Clinical Decision Support.* BMJ Health & Care Informatics, 2026; 33(1): e101877. DOI: 10.1136/bmjhci-2025-101877.
2. Cajas Ordóñez SA, et al. *Beyond Overconfidence: Embedding Curiosity and Humility for Ethical Medical AI.* PLOS Digital Health, 2026; 5(1): e0001013. DOI: 10.1371/journal.pdig.0001013.
3. Li SS, Balachandran V, Feng S, Ilgen J, Pierson E, Koh PW, Tsvetkov Y. *MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning.* NeurIPS 2024.
4. Ravenda F, et al. *Hx2Dx: From Anamnesis to Diagnosis — A Multimodal Arena for Conversational Clinical Reasoning in LLMs.* arXiv:2611.14122.
5. Braitsch K, et al. *Information-seeking failures of large language models in agentic clinical reasoning.* arXiv:2607.10275.
6. Arora R, et al. *HealthBench: Evaluating Large Language Models Towards Improved Human Health.* arXiv:2505.08775.
