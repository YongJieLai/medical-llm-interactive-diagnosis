# CEV-GR 底座移植性分析（react_cv / react_cv2 / roleplay_cv）

> 结论一句话：**CEV-GR 是"采集-诊断解耦"底座（sc / refine / roleplay）的可插拔审计层，
> 不是全范式通用插件。** ReAct（react）底座与它在结构上不兼容——无论是通用移植
> （react_cv）还是专用移植（react_cv2），准确率都掉 6.5～8pp。

---

## 0. 一句话背景

上一轮把 CEV-GR 四检查门控通用移植到 ReAct 底座，得到 `react_cv`，结果 46.5%（react 原样
53.0%，**−6.5pp**）。本轮做三件事：① 分析 react_cv 掉点根因；② 针对 react 底座出专用
CEV-GR（`react_cv2`）；③ 补跑 `roleplay_cv`（另一个真正没装过插件的底座）。

## 1. 全量结果（空起步 · 200 例 · qwen-plus · 16 turns）

| 底座 | 原样 | +通用 CEV-GR | +react 专用 CEV-GR | Δ（最佳插件 vs 原样） |
|------|:---:|:---:|:---:|:---:|
| sc | 59.0% | refine_cv 58.5% | — | −0.5pp |
| refine | 55.0% | refine_cv 58.5% | — | +3.5pp |
| **roleplay** | 49.0% | **roleplay_cv 57.5%** | — | **+8.5pp** |
| **react (ReAct)** | 53.0% | react_cv 46.5% | **react_cv2 45.0%** | **−8.0pp** |

关键证据覆盖率（正确组 / 错误组，综合 information coverage）：

| 底座 | 原样 | +通用 CEV-GR | +react 专用 |
|------|------|------|------|
| react | 25.3% / 20.8% | 26.6% / 22.7% | 25.6% / 22.7% |
| roleplay | 33.5% / 31.3% | 34.6% / 31.1% | — |

## 2. react_cv 掉点根因（为什么通用移植失败）

### 2.1 反直觉事实：覆盖率没变差，甚至略升

| 指标 | react | react_cv |
|------|:---:|:---:|
| 正确组综合覆盖 | 25.3% | 26.6% |
| 错误组综合覆盖 | 20.8% | 22.7% |

GATHER 反馈确实让医生多采了一点证据。**掉点不在"采集"，在"诊断决策被破坏"。**

### 2.2 真正的杀手：give_up 从 0% 涨到 13.2%

| 指标 | react | react_cv |
|------|:---:|:---:|
| 错题 give_up（不敢下诊断） | 0/94 = 0% | 14/106 = 13.2% |
| 平均 turns | 8.5 | 11.6 |

通用 CEV-GR 丢弃了 ReAct 医生自己的 `[DIAGNOSIS]`，改用**独立 diagnostician** 重新诊断。
ReAct 医生采集的证据本来就浅（8.5 turns 就诊断，覆盖率仅 25%），独立 diagnostician 在这个
浅 summary 上有 14 例直接放弃（"insufficient / pending evaluation"）。这是 ReAct 医生自己
永远不会做的（它总是给诊断）。

### 2.3 三个机制根因

1. **`[DIAGNOSIS]` 答案被丢弃**：react_cv 的最终答案来自独立 diagnostician，不是 ReAct 医生
   自己的诊断 → 把一个 53% 的诊断器换成了在浅 summary 上只有 46.5% 的链条。
2. **`[THOUGHT]` 推理丢失**：`dialogue_history` 只记 `action_content`，ReAct 医生的推理链
   没进证据态。
3. **feedback 用 SC 格式**：`"your NEXT action must be [QUERY]/[TEST]"` 与 ReAct 的
   "Reasoning-then-Acting + [DIAGNOSIS] 是终点"契约冲突。

## 3. react 专用 CEV-GR（react_cv2）：仍失败（45.0%）

针对 2.3 三点，react_cv2 做了三处修正：

1. **候选诊断 = ReAct 医生自己的 `[DIAGNOSIS]`**（不引入独立 diagnostician）；
2. **evidence_state = 完整对话历史（含 `[THOUGHT]`）**；
3. **verifier 用 method-agnostic DiagnosticGate**（自己提取 key findings，适配无 `[EVIDENCE]`
   结构）+ feedback 改写成 ReAct 契约（REDO→`[THOUGHT]+[DIAGNOSIS]`；GATHER→`[THOUGHT]`+一个
   `[QUERY]/[TEST]`）。

结果：**45.0%，比 react_cv（46.5%）还低 1.5pp，比 react 原样低 8pp。**

| 指标 | react | react_cv | react_cv2 |
|------|:---:|:---:|:---:|
| 准确率 | 53.0% | 46.5% | 45.0% |
| give_up（错题） | 0% | 13.2% | 0% |
| 平均 turns | 8.5 | 11.6 | 10.6 |
| 正确组综合覆盖 | 25.3% | 26.6% | 25.6% |

**结论：give_up 不是主要杀手。** react_cv2 把 give_up 压回 0%，准确率仍更低——因为 gate 的
REDO / BREAK_ANCHOR / GATHER 让 ReAct 医生"反复横跳"，最终诊断偏离它一次到位的判断，且
多问的 2 轮没有采到更多关键证据（覆盖率 25.6% 与 react 25.3% 几乎持平）。

### 3.1 为什么 react 底座与 CEV-GR 结构性不兼容

不是"格式不匹配"（通用 vs 专用都失败），而是三个**结构**性冲突：

1. **`[DIAGNOSIS]` 是"不可逆终点"**：ReAct 医生把诊断当结论，被 gate 打回后行为退化，无法
   进入"验证→回头采集"的循环；
2. **效率优先的浅采集**：ReAct prompt 第 1 条就是 "gather in as few turns as possible"，
   8.5 turns 就诊断，覆盖率 25%，比 roleplay/sc/refine 的 33–35% 低一大截；CEV-GR 的
   gap-driven 补齐在这个浅底子上救不回来；
3. **采集-诊断单 agent 耦合**：ReAct 医生同时负责采集和诊断，gate 干扰诊断会连带破坏采集
   策略。而 sc/refine/roleplay 是"采集器 + 独立诊断器"解耦，gate 只作用于诊断器，不伤采集器。

## 4. roleplay_cv：大胜（57.5%，+8.5pp）

roleplay 底座（base doctor `[QUERY]/[TEST]/[DIAGNOSIS]`）与 sc/refine 同构——医生是采集器，
诊断交给独立 diagnostician。通用 CEV-GR 直接有效，且因为 roleplay 原样的诊断器最弱（49%），
四检查补短板的空间最大：

| 指标 | roleplay | roleplay_cv |
|------|:---:|:---:|
| 准确率 | 49.0% | **57.5%（+8.5pp）** |
| 正确组综合覆盖 | 33.5% | 34.6% |
| 平均 turns | 12.9 | 13.9 |

这印证了 CEV-GR 的适用边界：**底座越"采集-诊断解耦"，插件提升越大；底座越"采集-诊断耦合 +
诊断=终点"，插件越伤。**

## 5. 对论文定位的最终修正

| 原表述 | 修正后 |
|------|------|
| CEV-GR 是零训练通用诊断插件 | CEV-GR 是**采集-诊断解耦底座的可插拔审计层**（sc/refine/roleplay 系） |

- **适用底座**：sc（59→58.5，持平）、refine（55→58.5，+3.5）、roleplay（49→57.5，+8.5）。
- **不适用底座**：react/ReAct（53→46.5/45，−6.5～−8）。
- **CEV-GR 的真实支点仍是关键证据覆盖率（正确组 52–62% vs 外部方法 12–22%）与错题根因结构
  （跨过采集关）**，而不是 top-1 准确率本身。

## 6. 工程记录

- `scenarios/react_cv2.py`（新建）：`ReactCV2Scenario`，ReAct 底座 + method-agnostic
  DiagnosticGate + ReAct 契约 feedback。
- `scenarios/__init__.py`、`src/run_paper_modes.py`：注册 `react_cv2`。
- 实验：react_cv2 / roleplay_cv 各 200 例 · 16 turns · 8 片分片 · workers=3。
- 判分：`deterministic_judge`（规则 + DeepSeek 双判兜底），统一口径。
