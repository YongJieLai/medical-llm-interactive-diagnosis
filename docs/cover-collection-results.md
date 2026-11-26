# 方案 A 结果：广覆盖采集（去「效率优先」）——首个正结果

> 状态：**已完成，正结果**（2026-11-27）
> 前置：`main-battlefield-cover-driven-collection.md`（主战场=采集层，方案 A 是最小改动）。
> 本文记录方案 A（`refine_cv_cover`）的实现与 200 例实测结果。

## 0. 一句话结论

**主战场假设被验证：采集层的"效率优先"是覆盖率低的元凶。** 去掉它 + 注入 8 维结构化证据框架后，**覆盖率中位从 31% → 50%（+19pp），准确率从 58.5% → 65.0%（+6.5pp，净 +13 例）**。这是全项目在采集层拿到的第一个正结果，直接命中"覆盖率中位 45–50%"的目标区间。

## 1. 改动内容（纯 prompt、零额外调用、零训练、不泄题）

`benchmarks/eid-benchmark/src/eid/scenarios/refine_cv_cover.py`（`RefineCVCoverScenario`）：

1. 把 doctor system prompt 第 1 条
   > "Efficiency: Gather sufficient information in **as few turns as possible**."
   改为
   > "Comprehensiveness: Gather a **COMPLETE evidence picture** across all eight domains below. Do NOT finish early just to save turns."
2. 注入 8 维结构化证据框架（主诉/现病史/伴随症状/既往史/用药史/社会与职业暴露史/查体/实验室与影像），作为"关键证据类别"的候选空间。
3. 其余四检查门控（claim/gap/refutation/decisive）**原样保留**——只改采集层，验证"单点松绑覆盖率"的效果。

## 2. 实测结果（200 例 / 16 turns / qwen-plus，空起步，同判分口径）

| 指标 | refine_cv（基线） | refine_cv_cover（方案A） | Δ |
|---|---|---|---|
| **准确率** | 58.5%（117/200） | **65.0%（130/200）** | **+6.5pp** |
| **覆盖率中位** | 31.0% | **50.0%** | **+19.0pp** |
| 覆盖率均值 | 32.4% | 50.1% | +17.7pp |
| **覆盖率 ≥50% 的 case** | 7.5%（15/200） | **51.5%（103/200）** | **+44.0pp** |
| 正确组覆盖率 | 34.3% | 53.7% | +19.4pp |
| 错误组覆盖率 | 29.7% | 43.4% | +13.7pp |

**配对显著性（McNemar）**：both对=93，基线对/cover错=24，基线错/cover对=37，both错=46。
χ²=2.36，**p=0.124（未过 0.05，但方向明确、幅度可观：修好 37 例 vs 改坏 24 例，净 +13 例）**。

## 3. 怎么读这个结果

1. **主指标（覆盖率）的提升是无争议的、决定性的**：中位翻倍（31%→50%），≥50% 段 case 从 15 个涨到 103 个（6.9 倍）。而覆盖率是贯穿全项目唯一"能分辨方法好坏"的强信号（正确组 vs 错误组差 +9~18pp 全显著）——方案 A 在这里 +19pp，远超任何此前的尝试。
2. **准确率 +6.5pp 是实质收益，但尚未过显著性阈值**（p=0.124）。诚实地说：方向确定为正，幅度可观，但需要更大样本或与方案 B 叠加后再定显著性。
3. **"覆盖够了仍错"的 case 暴露了剩余瓶颈在诊断层**：覆盖率 ≥50% 段的准确率从基线的 86.7%（13/15）降到 73.8%（76/103）。这不是退化——基线那 15 个 case 是"少数幸运采够的易题"，方案 A 把 103 个 case 硬抬到 ≥50%，其中一部分**证据够了但诊断器仍然错**（overdiagnosis 等偏置）。这正好把层 2（诊断层去偏置）的作用对象显性化了。

## 4. 结论与下一步

- **方案 A 成立**：主战场的锚点转换（效率优先 → 广覆盖证据框架）有效，覆盖率的瓶颈确实是"采集期的效率偏好"。
- **主战场尚未打通**：覆盖率上来了，但准确率只 +6.5pp（未显著），且 ≥50% 段还有 26% 的 case 诊断错。剩余空间拆成两块：
  1. **采集的"质"**：错误组覆盖率仍只有 43.4%（vs 正确组 53.7%），且"广覆盖"采到的是"量"不一定是"关键证据"——这是方案 B（证据类别缺口驱动的定向采集）要解决的：把覆盖从"均匀撒网"升级为"关键缺口优先"。
  2. **诊断的"偏置"**：层 2 的 base-rate prior（治 overdiagnosis 14–16%），在覆盖率不再是瓶颈后，它成为更清晰的独立杠杆。

**下一步（按顺序）**：
1. **方案 B**：在方案 A 基础上加"证据类别缺口"采集规划器（前移 gap check），对照 A vs A+B 的覆盖率与准确率增量。
2. **层 2 去偏置**：在 A+B 底座上叠加 diagnostician 的 base-rate prior，看 overdiagnosis 是否下降。
3. 三者叠加若达 70%+，即形成完整方法（覆盖采集 + 缺口定向 + 去偏置诊断）成文。

## 5. 落地清单

- 代码：`benchmarks/eid-benchmark/src/eid/scenarios/refine_cv_cover.py`（`RefineCVCoverScenario`）
- 注册：`scenarios/__init__.py`、`src/run_paper_modes.py`、`src/summarize_paper_modes.py`、`src/analyze_coverage.py`、`src/analyze_key_facts.py`
- 实验：200 例 / 16 turns / qwen-plus / 10 workers，57.8 分钟跑完
- 判分/覆盖率：`analyze_coverage.py`（deterministic_judge + coverage）
