# 医学 LLM 诊断：抗幻觉 / 证据锚定 / 假设管理 —— 相关工作梳理

> 调研目的：不闭门造车。我们做插件迭代时在 AgentClinic 场景实测出三类失效根因，
> 本文件把「别人是怎么做的」这条线摸清楚，并对照我们的做法，给出可零训练落地的下一步。

## 一、为什么调研这条线（我们实测出的三个失效根因）

在 30 例 AgentClinic 诊断实验（refine 基线，qwen-plus）中，确定性 judge 复核后定位到三个真实失效根因：

| 病例 | 真值 | 根因 | 文献中的对应分类 |
|------|------|------|-----------------|
| medqa_16 | C. difficile colitis | **幻觉阴性检查**：病例里 CDI toxin 是"Not specified"，模型写成"tcdB PCR negative"并据此排除 CDI | 灾难性幻觉（catastrophic，编造排除性证据） |
| medqa_14 | 屈指深肌腱断裂(Jersey finger) | **方向性体征读反**：病例写"无法**屈曲** DIP"，模型当成"无法伸直"→mallet finger | 关键性幻觉（critical，误判疾病类型） |
| medqa_15 | Epidermoid cyst | **关键鉴别证据未利用**：明写"囊性 + 皮肤无法捏起"，模型忽略选 Lipoma | 假设管理/DDx 维护不足 |

这三类正好对应文献里的三条技术路线：**证据锚定/抗幻觉**、**假设管理/DDx 维护**、**多 agent 互审/论证**。

---

## 二、三条技术路线与代表工作

### 路线 A：证据锚定 / 抗幻觉（对治"幻觉阴性检查""读反"）

**A1. Chain-of-Verification（CoVe）— Dhuliawala et al., ACL 2024 Findings**
- **机制**：四阶段——①生成 draft ②规划"验证问题"清单 ③**独立**回答每个验证问题（不看原始 draft，防止自我合理化）④对照验证结果修订最终答案。
- **核心洞察**：CoT 只改善"逻辑"，不检查"事实"；CoVe 把初始回答当"草稿"，逐条核对事实。
- **关键点**：第③步必须"独立验证"——这是防止模型"自圆其说"的关键（我们 HPR 的 verifier 恰恰少了这一步：它拿到了诊断和反证，却用"clinical misstatement"来自我合理化）。
- **临床类比**（文献转述）：住院医给方案 = draft；主治逐条质疑"肾功能多少？体重对不对？"= 独立验证。

**A2. MedCite — Wang et al.（Can Language Models Generate Verifiable Text for Medicine）**
- **机制**：多趟检索 + 语句级（statement-level）引用。先把答案拆成独立语句，每条语句重新检索最相关的证据（BM25 词法 + MedCPT 语义 + LLM rerank），强制每条临床声明有可验证引用。
- **关键发现**：①**参数化引用（自引用）不可靠**——让模型凭内部记忆引文献会编造 URL/论文，必须非参数化外部检索；②词法检索（BM25）在医学里往往优于纯语义检索（精确术语是强信号）。
- **与我们的关系**：我们抓到的"幻觉阴性检查结果"就是典型的"自引用编造证据"。MedCite 的解法是"逐句核对证据来源"，零训练版可简化为"诊断的每条支持证据必须回指 summary 原文"。

**A3. A2R（Ask, Assess, Refine）— Lee et al.**
- 用 citation-aware 指标评估"引用错误"与"幻觉"，改善医学文档的事实一致性。

**A4. Self-reflection / verification 框架族（系统综述归纳）**
- 综述 [PMC13470931] 把医学抗幻觉方法归纳为：RAG、KG 增强推理、KG 验证、self-reflection/verification（10 篇）。
- 代表：Factual Knowledge Acquiring Loop、Knowledge-Consistent Answering Loop、**CoMT（Chain-of-Medical-Thought，灾难性幻觉减少 38%）**。
- **MediHall Score 分级**（关键评估工具）：把幻觉按临床影响分三级——**catastrophic**（漏诊/编造疾病，0 分）、**critical**（误判疾病类型，0.3 分）、**attribute**（误判形状/大小/位置，0.6 分）。我们的 medqa_16（编造 CDI 阴性）= catastrophic，medqa_14（flex/extend 读反）= critical。

**A5. Semantic entropy — Farquhar et al.**
- 零训练、模型无关的幻觉探测：采样多个回答→语义聚类→熵高则更可能错。可作"选择性作答/转人工"的安全门。

---

### 路线 B：假设管理 / DDx 维护 + 信息获取（对治"关键鉴别证据未利用""锚定"）

**B1. MedKGI（Iterative DDx with Medical KGs and Information-Guided Inquiring）**
- **三大组件**：①**KG 锚定假设生成**（PrimeKG，约束在已验证的疾病-症状关系上）②**信息增益提问**（用 Shannon 熵挑"最高区分度"的下一问）③**OSCE 结构化状态记录**（JSON 跨轮维护已确认症状/病史/检查，防上下文过载）。
- **关键机制（直接相关）**：**"若候选集连续多轮不变，提示 agent 寻找能反驳当前假设的证据"**——这就是我们 HPR 想做但没做对的"反证复盘"，但 MedKGI 把它做成确定性触发，而非让 LLM 自由发挥。
- **消融**：去掉 KG 掉 25.47%；agent-MedQA 69.81%（Qwen3-8B）、agent-CMB 68.21%、NEJM 多模态 69.09%，对话轮数比基线少约 30%。

**B2. Ontology-guided Multi-Agent DDx（neuro-symbolic）**
- LLM 做语言理解 + 本体（SYMP/DOID/HPO）做符号推理：**数学打分函数**给诊断排序、每步可追溯；用 KG 找"最能区分竞争诊断的缺失症状"生成定向追问。
- **关键主张**：把"语言理解"和"临床推理"解耦，推理落在可审计的符号层，减少幻觉、增加可解释性。

**B3. 信息引导提问族**：UoT（Uncertainty of Thoughts）、PATIENCE（Bayesian active learning）、MedAgent（多 agent 协作）。

---

### 路线 C：多 agent 互审 / 形式论证（对治"自我合理化""一致性"）

**C1. MAP（Multi-Agent Inpatient Pathways）— Zhen Chen et al.**
- 临床层级多 agent（Lead/Diagnostic/Safety Auditor）互相 cross-verify，抓复杂工作流里的逻辑疏漏。
- **核心观点**：单模型难自我纠错，因为"导致幻觉的统计偏差同时让它看不见自己的错"——所以需要**分布到多个专门 agent 互相审计**。

**C2. Argumentation-driven framework — Rodrigo Agerri et al.**
- 模型必须**用已验证的临床证据为诊断辩护、并反驳备择假设**；强制把输出组织成"可辩护的论证"，暴露逻辑缺口和幻觉。

---

## 三、我们的插件 vs 文献对照（关键认知修正）

| 维度 | 我们的第一代 HPR | 文献的做法 | 差异与教训 |
|------|----------------|-----------|-----------|
| 假设保护 | 加 differential 生成 top-5 DDx 列表，diagnostician 从列表选 | MedKGI/ontology MAS 用 **KG/本体** 锚定候选 + **数学打分**排序 | 我们让"列表"本身成了决策者，而列表生成有随机性→回归。文献用**确定性锚（KG/打分）**而非"另一个 LLM 生成列表" |
| 反证复盘 | verifier 一致性检查 | MedKGI 的**确定性触发**（候选不变→强制找反证）；argumentation 的**强制辩护+反驳** | 我们的 verifier 识别了矛盾却用"clinical misstatement"自我合理化；文献用**独立核对 + 强制反驳**避免自我合理化 |
| 证据锚定 | diagnostician prompt 加三条规则 | CoVe 的**独立验证步骤**；MedCite 的**语句级引用回指** | 我们只是"同一 prompt 里自我约束"；文献强调**拆出独立核对步骤**，这是关键差别 |

**一句话总结教训**：把约束写进同一个 LLM 的 prompt 里，模型会用流畅的"自我合理化"绕过约束；文献的共识是**把生成与核对解耦**（独立验证 / 独立 agent / 确定性符号锚）。

---

## 四、可零训练落地的下一步（从文献提炼，按优先级）

1. **CoVe 式独立核对（最优先）**：diagnostician 出诊断后，加一个**不看诊断推理过程**的独立 verifier，任务不是"判 PASS/INCOMPLETE"，而是"把诊断里每个证据声明逐条和 summary 核对，标出【未在 summary 出现 / 与 summary 矛盾】的声明"。直接对治 medqa_16 式幻觉阴性、medqa_14 式读反。
2. **语句级证据回指（MedCite 零训练版）**：诊断必须附"支持证据清单，每条回指 summary 原句"；缺证据/编证据当场暴露。
3. **确定性反证触发（MedKGI 版）**：诊断在多轮不变时，强制 verifier 输出"当前诊断的 1 条反证 + 1 个未排除的鉴别诊断"，而非自由发挥。
4. **MediHall Score 分级记分**：把 judge 的"对/错"升级为"灾难性/关键性/属性级"三级，让失效分析更精细（我们已发现 catastrophic 与 critical 两类）。

---

## 五、参考文献

1. Dhuliawala et al. *Chain-of-Verification Reduces Hallucination in LLMs*. Findings of ACL 2024. 机制说明见 https://rruc.org/chain-of-verification-cove-how-to-stop-llm-hallucinations-in
2. Wang et al. *MedCite: Can Language Models Generate Verifiable Text for Medicine?* — https://lacuna.tiptreesystems.com/work/medcite-can-language-models-generate-verifiable-text
3. *Mitigating hallucinations in healthcare AI: a systematic review* — https://pmc.ncbi.nlm.nih.gov/articles/PMC13470931/ （含 MediHall Score、CoMT、A2R、self-reflection 族）
4. *Mitigating Hallucination Risks in Clinical Language Models*（MAP、argumentation、MedCite、MHB）— https://lacuna.tiptreesystems.com/direction/medical-hallucination-risk/rdr_87ffbaabaee742baab43f8d9c118bc7e
5. *MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring* — https://arxiv.org/pdf/2512.24181v2
6. *Ontology-guided multi-agent reasoning for explainable differential diagnosis* — https://arxiv.org/html/2610.36325v1
7. Farquhar et al. *Detecting Hallucinations in LLMs using Semantic Entropy*（综述见 https://pmc.ncbi.nlm.nih.gov/articles/PMC12429116/）
