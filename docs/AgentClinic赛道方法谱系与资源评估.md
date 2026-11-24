# AgentClinic 赛道方法谱系与资源评估

> 目标：不刷 SOTA，而是在 AgentClinic/MedExAgent 这类难环境里跑「最新推理时方法」，定位失效场景，为后续提「零训练可插拔插件」积累证据。
> 调研时间：2026-11-24

---

## 一、赛道全貌：两代方法已经换代

之前选的 MedKGI / SGUR / CDEG / KnowGuard 属于「信息增益选问 + KG」那一代（跑在 DDXPlus 这种**结构化证据表**环境上）。AgentClinic/MedExAgent 赛道的主流已经换成 **「多智能体编排 + 记忆 + 树搜索 + 工具使用」**。

| 派别 | 代表 | 一句话 | 我们能不能跑 |
|---|---|---|---|
| **训练派**（SOTA 天花板） | Cura 1T、MedExAgent、DoctorAgent-RL | 把「会问诊」炼进参数，8B 也能匹配 70B | ❌ 不跑，只当参照 |
| **推理时派**（零训练、可插拔） | REFINE、ClinicalAgents、MEDDxAgent、原版 6 工具 | 不改参数，靠脚手架/记忆/树搜索 | ✅ **这就是我们要跑和要做的方向** |

**AgentClinic-MedQA 成绩单**（越低越接近真实难度）：

| 主体 | 分数 | 性质 |
|---|---|---|
| Cura 1T | **79.6** | 训练（1T 参数 + 递归自改进） |
| Claude Opus 4.8 | 79.4 | 闭源 frontier |
| MedExAgent-8B | ≈ 70B Aloe-Beta | 训练（POMDP + DAPO RL） |
| Claude 3.5 Sonnet（2024 原版） | 62.1 | 模型基线 |
| OpenBioLLM-70B | 58.3 | 医学微调 |
| 人类医师 | 54 | 人类基线 |

---

## 二、第三层三个方法详解（机制 + 资源需求 + 移植成本）

### 1. REFINE（EviMed，ACL 2026，arXiv 2601.19773）⭐ 首选

**机制**：三角色（医生 / 模拟患者 / 模拟 reporter，均基于原子证据）+ 「诊断验证」环——先出候选诊断 → 发现证据缺口 → 针对性补问，用 **ICR（信息覆盖率）** 单独量化「证据挖得全不全」。

**资源需求（实测 repo `NanshineLoong/EID-Benchmark`）**：

| 维度 | 结论 |
|---|---|
| 模型接入 | ✅ OpenAI 兼容（`.env` 配 `OPENAI_API_BASE_URL` + `OPENAI_API_KEY`），国产 API 直接接 |
| GPU | ❌ 不需要，纯 API |
| 数据集 | ✅ **内置 AgentClinic-MedQA**（`data/agentclinic_medqa_segmented.jsonl`）+ DiagnosisArena + RareArena + Derm(CRAFT-MD) |
| 外部知识库 | ❌ 不需要 |
| 移植成本 | 🟢 **几乎零改造**，`eid-eval --datasets medqa --modes roleplay/react/sc/refine --doctor-model <模型>` 即可跑 |

**关键**：这是三个里**唯一直接内置 AgentClinic 数据集**的，且 `refine` 模式就是论文方法本身，`roleplay`/`react`/`sc` 可直接当 baseline 对照。

---

### 2. ClinicalAgents（港理工，KDD 2026，arXiv 2603.26182）

**机制**：MCTS 编排 + 双记忆——工作记忆（患者状态 + 假设 + 轨迹）＋ 经验记忆（临床指南库 + 历史病例 CDC 库，经 AGRAG 检索），证据缺失就触发**结构化回溯**（缺影像回影像、缺病史回问诊）。

**资源需求（实测 repo `ZhuohanGe/ClinicalAgents-Code`）**：

| 维度 | 结论 |
|---|---|
| 模型接入 | ✅ OpenAI 兼容（`LLM_CONFIG` + `JUDGE_CONFIG`，base_url 可配） |
| GPU | ❌ 不需要，纯 API |
| 数据集 | ⚠️ 面向 **MedChain 五阶段流程**（含影像 `img_api`），不是 AgentClinic 纯问诊 |
| 外部知识库 | 🔴 **需要自备**：`EXPERIENCE_GUIDE_PATH`（临床指南库）+ `EXPERIENCE_CDC_PATH`（历史病例因果链库），**均不在 repo 里** |
| 配置文件 | 🔴 `app_config.py` 缺失，需自己建（LLM/JUDGE/路径常量） |
| 移植成本 | 🟠 中等：补配置 + 补知识库 + 把 MedChain 流程改造回 AgentClinic 纯问诊 |

**注意**：它论文里**没在 AgentClinic 上评测**（在 MedChain + 4 个静态 QA 上）。移植过来本质是「把它的 MCTS+双记忆脚手架搬上 AgentClinic 考场」。

---

### 3. MEDDxAgent（NEC，ACL 2025，arXiv 2502.19175）

**机制**：DDxDriver 调度器（ReAct 循环）＋ 病史采集模拟器（双 LLM：患者 + 医生）＋ 知识检索 agent（Wikipedia/PubMed）＋ 诊断策略 agent（零样本/少样本）。

**资源需求（实测 repo `nec-research/meddxagent`）**：

| 维度 | 结论 |
|---|---|
| 模型接入 | ⚠️ 原版依赖 **Azure OpenAI + vllm(Llama3.1-70B，需 GPU) + huggingface(Llama3.1-8B)**；但有 `oai_chat.py`（langchain ChatOpenAI）可改 base_url 接国产 API |
| GPU | 🟠 原版要 GPU（跑 70B）；**改纯 API 后可绕开** |
| 数据集 | ⚠️ 格式是 `initial profile / complete profile / GT`，内置 ICraftMD + RareBench，DDxPlus 从 HF 下载；**不是 AgentClinic 格式** |
| 外部知识库 | 🟠 检索 agent 需要 Wikipedia + PubMed（网络检索） |
| 移植成本 | 🟠 中等偏高：改模型层 + 改数据接口适配 AgentClinic |

**注意**：它论文里在 **DDxPlus/iCraftMD/RareBench** 上评测（+10%），不在 AgentClinic 上。

---

## 三、资源需求横向对比

| 维度 | REFINE ⭐ | ClinicalAgents | MEDDxAgent |
|---|---|---|---|
| 模型接入（国产 API） | ✅ 直接接 | ✅ 直接接 | ⚠️ 需改一层 |
| 需要 GPU | ❌ | ❌ | 🟠 原版要，可绕开 |
| 内置 AgentClinic 数据 | ✅ | ❌（是 MedChain） | ❌（是 DDxPlus） |
| 外部知识库 | ❌ | 🔴 指南库+病例库 | 🟠 Wiki/PubMed |
| 缺配置文件 | ❌ | 🔴 app_config 缺失 | ❌ |
| 移植工作量 | 🟢 零 | 🟠 中 | 🟠 中高 |

---

## 四、结论：几个 API 够不够？

**直接回答：够，而且一个就能起步；真正的瓶颈不是「API 数量」，是「API 调用量」和「外部数据工程」。**

### 1. API 数量层面 —— 一个就够，两三个更稳

三个方法全部 OpenAI 兼容（`base_url` 可配）。多智能体框架里的「医生/患者/reporter/裁判」只是**同一个 API 下的不同 model 名（或同一个 model）**，不需要多个账号。你手上的千问 / DeepSeek / GLM 任意一个都能当 backbone，另拿一个强模型当裁判（judge）即可。

### 2. 真正的成本在「调用量」，不是「数量」

| 方法 | 每病例 LLM 调用量级（估） | 说明 |
|---|---|---|
| REFINE | ~24–48 次 | 医生 N 轮 ×（医生 + 患者 + reporter） |
| ClinicalAgents | **~上百次** | MCTS 每步 N 次 rollout × 多 agent，最吃额度 |
| MEDDxAgent | ~数十次 | 调度器 + 双 LLM 问诊 + 检索 + 诊断策略 × 迭代 |

→ 建议：跑 ClinicalAgents 前先算好「病例数 × 单病例调用量 × 单次 token」，否则千问额度/限流会先崩。**先小规模（3–5 病例）验证单病例真实调用量，再放量。**

### 3. 需要的「API 之外」的资源

- **外部知识库**：ClinicalAgents 要临床指南库 + 历史病例 CDC 库（repo 里没有，得自己找或造）；MEDDxAgent 要 Wikipedia/PubMed 检索。
- **数据接口改造**：ClinicalAgents 面向 MedChain（含影像）、MEDDxAgent 面向 DDxPlus，搬上 AgentClinic 都要改。

---

## 五、建议的第一批实验节奏（第三层）

```
第 1 步（零改造、当天能跑）：REFINE
   千问当 backbone，跑 AgentClinic-MedQA（内置）
   modes = roleplay(下界) / react / sc / refine(论文方法)
   先 3–5 病例验证调用量 → 再 20–30 病例

第 2 步（补知识库后）：ClinicalAgents
   补 app_config + 指南库/CDC 库（或先造小规模子集）
   把 MedChain 流程改回 AgentClinic 纯问诊

第 3 步（改模型层+数据接口后）：MEDDxAgent
   改 oai_chat 接国产 API，改数据接口适配 AgentClinic
```

**一个值得记住的判断**：真正「在 AgentClinic 上系统跑最新推理时方法 + 失效分析」这件事目前没人做——原版只测了 6 个简单工具（2024 那代），ClinicalAgents/MEDDxAgent/REFINE 各自测在自己的 benchmark 上。把这几个 2026 年最新脚手架搬上 AgentClinic、看它们在 LLM 患者噪声 + 主动开检查 + 20 轮限制下怎么失效，正是你要做插件前的证据基础。
