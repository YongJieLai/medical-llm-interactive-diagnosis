# 方案 B 结果：采集规划器（前移 gap check）是负结果

> 状态：**负结果，已定位根因**（2026-11-27）
> 前置：`cover-collection-results.md`（方案 A 正结果）、`cover-collection-fairness.md`（方案 A 公平性审计）。
> 本文回答：在方案 A 上叠加"证据类别缺口采集规划器"（方案 B）是否进一步抬覆盖率与准确率。

## 1. 实验设计

| 场景 | 静态 8 维框架 | 每轮动态缺口规划 | 变量 |
|---|---|---|---|
| 方案 A（refine_cv_cover） | ✓ | ✗ | — |
| 方案 B（refine_cv_cover_plan） | ✓ | ✓ | 采集规划器 |

规划器（DeepSeek，跨模型解耦）每轮读当前对话，输出 1–3 条 `{category, question, why}`，
category 被白名单约束在 8 维框架内；下一轮 doctor 的 instruction 注入这份缺口清单。

**与 P0 的刻意区别**：清单锚点是「证据类别缺口」（哪个 domain 还没问到），不是
「区分 top-2 假设的检查」；prompt 明确要求"prioritise questions that ESTABLISH or
REFUTE a diagnosis, NOT discriminate"。

## 2. 结果（199 例配对，排除 1 例内容审核失败 medqa_82）

| 模式 | 准确率 | 覆盖率中位 | ≥50% case | 正确组覆盖 | 错误组覆盖 | 正确−错误质差 |
|---|---|---|---|---|---|---|
| 基线 refine_cv | 58.8% | 31.0% | 7.5% | 33.3% | 30.3% | +3.0% |
| 消融 freebudget | 57.3% | 37.1% | 21.6% | 40.0% | 33.3% | +6.7% |
| 方案 A cover | **64.8%** | 50.0% | 51.8% | 54.3% | 40.0% | **+14.3%** |
| 方案 B cover_plan | 59.8% | 50.0% | 51.3% | 51.9% | 48.3% | +3.6% |

**方案 B 净效应（cover_plan − cover）= −10 例**（McNemar p=0.221，未过显著阈值，
但方向为负且幅度与方案 A 的正方向相反）。

## 3. 三个关键信号

1. **覆盖率中位持平（50%→50%）**：规划器没有让 doctor 采到更多关键证据。
2. **准确率 −10 例**：规划器反而干扰了 doctor 原本（方案 A 框架下）已不错的采集。
3. **错误组覆盖率升（40%→48.3%）但准确率降**：这是决定性信号——规划器让 doctor
   在**错误的方向上问得更多了**，采到的是"非关键"甚至"误导性"证据，而不是能
   纠正诊断的关键证据。

## 4. 根因：category 是对的，question 滑回了 discriminate

规划器的**结构**（证据类别缺口）是对的，但它的**内容**（具体 question）在实际
输出时又滑回了 discriminate。smoke 输出即为直接证据：

> `- [associated_symptoms] Are you experiencing any shortness of breath…`
> `  (Distinguishes pneumonia from bronchitis and other respiratory infections)`

尽管 prompt 明确写了 "NOT … discriminate between two candidate hypotheses"，
DeepSeek 的 question 仍以"区分肺炎 vs 支气管炎"为措辞。**"证据类别"只是壳，一旦
落到"具体问题建议"，LLM 就重新锚回了假设区分。**

这与 P0、两阶段、freebudget、外部三方法（MedClarify/MEDIQ/UoT）是**同一个根因**
：凡是以"建议医生问什么具体问题"形式介入采集的外部信号，都会把采集锚回 discriminate，
在证据稀疏时被初始假设误差放大。

**而方案 A 之所以成功，恰恰因为它不提供具体问题建议——只给"该覆盖哪些类别"的
静态框架，把"问什么具体问题"的决策权留给 doctor 自己（在框架约束下天然保持 open）。**

## 5. 对主战场论点的精确定位

方案 B 的负结果把"关键证据识别"的难点精确到了**问题层**：

> - **类别层（哪个 domain 缺失）**：容易，静态框架即可解决（方案 A 已证明）。
> - **问题层（在该 domain 下提出不区分、只确立/推翻诊断的具体问题）**：极难，
>   任何让 LLM 生成"具体问题建议"的尝试都会滑回 discriminate。

## 6. 下一步：方案 B′（category-only 规划）

干净消融，把"缺口识别的价值"与"问题生成的害处"分开：

- **方案 B′**：规划器只输出 `{category, why}`（该类别为何关键），**不输出 question**；
  "问什么"完全留给 doctor。对照 A vs A+B′。
- **判定**：
  - B′ 若回升到 ≈A → "缺口识别"有用，害处全在"具体问题建议"；论文可落
    "coverage framework（无 question）"这一可迁移结论。
  - B′ 若仍持平/掉点 → 动态缺口提示本身无用，方案 A 的静态框架即采集层天花板，
    主战场应转向诊断层去偏置（层 2）或覆盖率终止（方案 C）。

## 7. 代码落点

- 实现：`benchmarks/eid-benchmark/src/eid/scenarios/refine_cv_cover_plan.py`
  （`RefineCVCoverPlanScenario`，override `_after_dialogue_turn` + `_decorate_doctor_instruction`）
- 注册：`scenarios/__init__.py` + `src/run_paper_modes.py`（mode `refine_cv_cover_plan`）
