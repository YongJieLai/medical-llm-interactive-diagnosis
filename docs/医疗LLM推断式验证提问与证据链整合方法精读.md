# 医疗 LLM「推断→验证式提问 + 证据链整合」方法精读

> 调研日期：2026-11-24
> 触发问题：每轮获取信息后做推理，得到"患者可能还有哪些症状"，用于避免重复问询、并把有用信息整合成证据链用于推理——目前有哪些工作是这条思路的？

---

## 0. 概念先纠正：三个易混概念

| 概念 | 含义 | 对任务目标的影响 |
|---|---|---|
| **跳过不问（演绎闭包）** | 推断出"可能有症状 X"→ 直接当成"有 X"，不再问 | ❌ 错误：推断≠确认，会漏掉真实信息 |
| **验证式提问（confirm/rule out）** | 推断出"可能有症状 X"（因怀疑病 Y）→ 问"你有没有 X"→ 用回答确认/排除 Y | ✅ 正确：推断是"待验证假设" |
| **证据链整合** | 把多轮确认的信息显式累积成结构化证据，用于最终推理 | ✅ 正确：是最终目标 |

> **任务目标的正确定义**：问出患者所有信息 + 把有用信息整合成证据链用于诊断推理。
> 推断的作用是 **①补进证据链辅助推理、②让下一轮提问聚焦于鉴别性信息**，而不是替代询问。

---

## 1. 文献里的精确对应物

这条思路在文献里有两个精确名字：

1. **hypothesis-driven clarifying questions**（假设驱动的验证式提问）
2. **decision-critical evidence acquisition / reappraisal**（决策关键证据的获取与重估）

最直接命中的两篇：

| 论文 | 命中点 | 关键机制 |
|---|---|---|
| **KGCQ / Thinking Like a Doctor**（MIT, TACL 2025） | 推断→验证式提问 | HG 猜候选疾病 → 3-hop 子图 → HV "confirm or rule out specific attributes" |
| **CDEG**（arXiv 2608.22899） | 证据链 + 缺失证据获取 | 成功/失败轨迹对比挖 Missing/Ignored/Blocking 证据 → 反事实验证 → 证据状态驱动干预 |

---

## 2. 精读一：KGCQ（Thinking Like a Doctor）

### 2.1 三步循环

```
对话历史 H_t
   ↓
[Step 1] HG 假设生成：Qwen-2.5-7B + 线性分类头 → 338 疾病概率 → top-n 锚点疾病
   ↓
[Step 2] 3-hop 子图提取：
   1-hop = 锚点疾病的属性（symptom/cause/risk_factor）
   2-hop = 共享这些属性的竞争疾病
   3-hop = 竞争疾病的属性
   ↓
[Step 3] HV 验证：子图线性化（"疾病A导致症状B"）→ 生成 CQ 确认/排除特定属性
   ↓
患者回答 → 更新 H_{t+1} → 回到 Step 1，直到证据充分输出诊断
```

### 2.2 核心原文（验证式提问，不是跳过）

> "If the HV judges the accumulated evidence insufficient, it generates a CQ grounded in 𝒢_sub to **confirm or rule out specific attributes**."

> "By exposing both shared and disease-specific attributes, this structure enables the system to generate CQs that either **strengthen confidence in a plausible hypothesis or rule it out by probing discriminative attributes**."

### 2.3 关键数字

| 模型 | Recall@1 | Turns |
|---|---|---|
| Claude-3.5-Sonnet | **0.267** | 9.4 |
| SFT (Qwen-2.5-7B, KGCQ) | 0.250 | **6.9** |
| GPT-4.1-mini | 0.243 | 14.1 |
| Qwen-2.5-7B (base) | 0.173 | 12.4 |
| CoD (adapted) | 0.106 | 2.0（过早诊断） |

- HG 分类头 Recall@1=0.142 → 完整系统 0.250，说明"子图推理纠正假阳性"贡献显著。
- 消融：纯参数知识 0.142 → +KG 0.218 → +KG+HG 0.243。

### 2.4 局限（批判）

1. **SFT 训练**，不是零训练（HG 分类头 + HV 微调）。
2. **DKG 极小**：338 疾病 / 847 症状，来自一本韩国 CPX 备考教材（10 Minutes CPX）。
3. **训练对话是 GPT-4o-mini 合成的**，且 HV 训练时用了 oracle subgraph + ground-truth 标签（teacher forcing 偏差）。
4. **MIMIC-IV 仅保留 28%** 档案，覆盖偏差。
5. **患者是模拟器**，非真实对话。
6. **绝对准确率极低**：最好 0.267，交互式诊断天花板未破。

---

## 3. 精读二：CDEG（Counterfactual Diagnostic Evidence Graph）

### 3.1 Motivation：诊断错误的两大来源

> **88.03% = 50.43%（关键证据没获取）+ 37.60%（证据拿到了但没利用）**

（审计 117 条失败轨迹：Incomplete 50.43% / Sufficient-but-unused 37.60% / Conflicting 11.97%）

### 3.2 三类决策关键证据

| 类别 | 集合运算 | 诊断学含义 |
|---|---|---|
| **Missing** | 成功轨迹有、失败轨迹没有 | 支持目标诊断但未被获取的关键证据 |
| **Ignored** | 两边都有、但结果不同 | 已获取但未被充分利用的证据 |
| **Blocking** | 只有失败轨迹有 | 支持源诊断、应阻止修订的证据 |

### 3.3 反事实验证（三个测试）

| 测试 | 干预 | 通过条件 |
|---|---|---|
| Completion | 把缺失证据加入失败轨迹证据集 | 验证器偏好从 D_s 翻转到 D_t |
| Reappraisal | 保持证据不变，诊断比较时显式暴露该证据 | 验证器偏好翻转 |
| Blocking | 把阻断证据加入成功轨迹证据集 | 验证器偏好从 D_t 翻转到 D_s |

> 验证器 = GPT-5.5；诊断顺序反转重复两次，两次都翻转才保留。
> **移除反事实验证：准确率 53.00% → 43.50%**（比 Flat Experience 48.50% 还低）——证明反事实验证是核心，不是图结构本身。

### 3.4 推理时的"选择性干预"（关键：跳过 vs 获取 vs 重估）

| 证据状态 | CDEG 行为 |
|---|---|
| 支持当前诊断 D_s 的证据**已观察到** | **跳过，不干预**（修订不适用，放行最终诊断）|
| 支持 D_t 的证据**缺失**，且有可用获取动作 | **提示执行对应的获取动作**（defer 最终诊断）|
| D_t 证据已全部拿到，但 agent 仍提议 D_s | **提示重估已观察证据**（不新增，改变利用方式）|

> 关键区分：CDEG 的"跳过"是"证据已足够所以不用再干预"，**不是"推断出症状就不问"**。它的"获取"是**触发图中已知的 acquisition-action 节点**，不是开放式生成追问。

### 3.5 关键数字

| 指标 | 数值 |
|---|---|
| MIMIC-IDx 平均提升 | +7.63% |
| AgentClinic 平均提升 | +8.88%（最大 +11.5%）|
| GPT-5.6-Luna @ AgentClinic | +11.22%（51.40→62.62），同时**减少 0.71 轮** |
| 额外交互轮数 | +1.41（ID）/ +1.27（OOD）|
| 图构建 | **零训练**（冻结共享），但用 GPT-5.5/Gemini 等 LLM 建图 |
| 图迁移性 | Gemini 轨迹建的图，对 GPT/Claude backbone 同样有效 |

### 3.6 局限（批判）

1. 图**静态**，不随推理在线更新（论文自认未来要"online graph evolution"）。
2. 验证器依赖 GPT-5.5，验证器自身误判未做消融。
3. 多个手工阈值（0.6/0.65/0.7/0.72）未做敏感性分析。
4. 图构建需**大规模 LLM 推理**，不是纯零训练（推理时是零训练）。

---

## 4. 其余同思路工作速览

| 工作 | 命中线 | 机制 | 训练/KG | 关键数字 |
|---|---|---|---|---|
| **KnowGuard** | 验证式提问 + 证据池 | 证据发现→五因素评估→不足弃权+针对性追问 | 依赖 300+ WHO 指南 KG（22k 节点）| +3.93% 准确率，token 省 95.7% |
| **GraphMed-LT** | 证据链整合 | 回答→三元组→增量图→证据 token 注入 | **训练**（可训练 doctor）| +6.3pp |
| **DeepRare**（Nature 2026，上交大+新华） | 两者都要 | 假设—验证—自我反思 + 主动提问 + 完整证据链 | 工业级罕见病专域 | 表型诊断 57.18%（+23.79pp），证据链报告医生认可 95.4% |
| **Counterfactual Multi-Agent**（arXiv 2603.27820） | 证据-假设关系量化 | 反事实编辑 + CPG 量化单 finding 支持度 | **零训练**，需 log-probs | 最大 +13.2pp，仅开源模型可用 |
| MedKGI / SGUR | 证据链整合 | OSCE 状态 / 增量查询图 | 零训练但强依赖 KG | 见前次精读 |

---

## 5. 四个批判性判断

1. **"验证式提问"与"证据链整合"在文献里是分开做的**，几乎没人把两者合起来：KGCQ 有验证提问但无显式证据链状态；CDEG 有证据链图但"获取"是触发已知动作而非开放式生成追问。

2. **绝对水平仍低**：KGCQ 最好 0.267、CDEG 提升后最好 62.62%。交互式诊断天花板（20–60%）未破，与 Hx2Dx/OncoRounds 结论一致。

3. **几乎都依赖 KG 或训练或大规模 LLM 建图**。真正"零训练 + 不依赖外部 KG"的方案在这条线里接近空白。

4. **CDEG 的反事实验证 ≈ 主动学习的"信息价值度量"**：加/去一个证据看诊断置信度是否翻转，本质就是"这个证据信息量多大"。这是主动学习背景可直接迁移的点。

---

## 6. 可卡位空白（结合零训练 + 主动学习背景）

> **零训练、不依赖外部 KG 的"推断→验证式提问 + 证据链显式累积"模块**

- 每轮用模型**内部知识**推断"候选诊断应有的待确认症状"（不用外部 KG）；
- 用主动学习式的信息价值度量（参考 CDEG 反事实验证：加/去某症状看诊断是否翻转）挑"最值得问的那个"；
- 问患者 confirm/rule out；
- 把确认结果作为证据节点**显式累积进证据链状态**。

**零件都有人单独做过**（KGCQ 的验证提问、CDEG 的证据链+反事实验证、MedKGI 的证据状态维护），但**拼在一起、且零训练去 KG**，目前无人占住。

---

## 7. 参考文献（待补 DOI 核对）

- KGCQ / "Thinking Like a Doctor"：TACL 2025，MIT（Holi Lab），code: github.com/holi-lab/KGCQ
- CDEG：arXiv 2608.22899
- KnowGuard：Lacuna（多模态 KG，300+ WHO 指南）
- GraphMed-LT：arXiv 2510.03536
- DeepRare：Nature 2026，上海交通大学 × 新华医院
- Counterfactual Multi-Agent Reasoning：arXiv 2603.27820
