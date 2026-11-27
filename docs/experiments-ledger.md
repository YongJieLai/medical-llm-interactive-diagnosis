# 实验结果总账（Experiments Ledger）

> 所有实验的统一索引，用于后续整理复盘。每条实验标记 **method（mode）/ 中文名 / 场景代码文件 / 结果文件 / 准确率**。
> 判分口径统一：`deterministic_judge`（规则层 + DeepSeek 兜底），MedQA 200 例，16 turns，doctor=qwen-plus（外部方法另有标注）。

---

## 0. 文件位置约定

| 内容 | 路径 |
|---|---|
| 交互 trace（原始） | `benchmarks/eid-benchmark/results/medqa/<mode>/qwen-plus/16_turns/record/<case_id>.json` |
| 逐例判分 | `results/coverage_analysis/percase_<mode>.json` |
| 覆盖率汇总 | `results/coverage_analysis/coverage_summary.json` |
| 关键事实标注 | `results/coverage_analysis/key_facts.json`（⚠️ 旧 confirmatory 定义，作废待重标） |
| 场景代码 | `benchmarks/eid-benchmark/src/eid/scenarios/<file>.py` |

---

## 1. 总表（按准确率降序）

| # | mode | 中文名/说明 | 场景代码文件 | percase/结果文件 | 准确率 |
|---|---|---|---|---|---|
| 1 | cot | CoT 静态全信息（上界） | `cot.py` | `percase_cot.json` | **94.0%** |
| 2 | refine_cv_cover | **方案A：去效率 + 8维框架** | `refine_cv_cover.py` | `percase_refine_cv_cover.json` | **65.0%** |
| 2a | roleplay_cover | 方案A移植：roleplay底座 + 8维框架 + CEV-GR | `refine_cv_cover_ports.py` | `percase_roleplay_cover.json` | **64.0%** |
| 2b | refine_cover | 消融：去效率 + 8维框架（无 CEV-GR） | `refine_cover.py` | `percase_refine_cover.json` | **60.0%** |
| 3 | refine_cv_qwen_no_decisive | CV同模型三检查 | `refine_cv_ablations.py` | `percase_refine_cv_qwen_no_decisive.json` | 60.0% |
| 4 | refine_cv_cover_plan | 方案B：A + 采集规划器 | `refine_cv_cover_plan.py` | `percase_refine_cv_cover_plan.json` | 59.8% |
| 5 | refine_cv_debias | 诊断层去偏置（分级强制） | `refine_cv_debias.py` | `percase_refine_cv_debias.json` | 59.8% |
| 6 | refine_cv_qwen_check | CV只 claim check | `refine_cv_ablations.py` | `percase_refine_cv_qwen_check.json` | 59.8% |
| 7 | refine_cv_p1 | P1：base-rate prior | `refine_cv_p01.py` | `percase_refine_cv_p1.json` | 59.0% |
| 8 | refine_cv_qwen_gap | CV只 gap check | `refine_cv_ablations.py` | `percase_refine_cv_qwen_gap.json` | 59.0% |
| 9 | sc | SC（总结诊断） | `sc.py` | `percase_sc.json` | 59.0% |
| 10 | refine_cv | CEV-GR 跨模型四检查 | `refine_cv.py` | `percase_refine_cv.json` | 58.5% |
| 11 | refine_cv_qwen_gap_refute | CV gap+refute（无 claim） | `refine_cv_ablations.py` | `percase_refine_cv_qwen_gap_refute.json` | 58.5% |
| 12 | refine_cv_cover_debias | 方案A + 去偏置（完整方法） | `refine_cv_debias.py` | `percase_refine_cv_cover_debias.json` | 58.0% |
| 13 | refine_cv_freebudget | 消融：只去效率、不加框架 | `refine_cv_freebudget.py` | `percase_refine_cv_freebudget.json` | 57.5% |
| 14 | refine_cv_no_decisive | CV跨模型三检查 | `refine_cv_ablations.py` | `percase_refine_cv_no_decisive.json` | 57.5% |
| 15 | refine_cv_qwen | CEV-GR 同模型四检查 | `refine_cv_qwen.py` | `percase_refine_cv_qwen.json` | 57.5% |
| 16 | roleplay_cv | roleplay 底座 + 四检查 | `refine_cv_ports.py` | `percase_roleplay_cv.json` | 57.5% |
| 17 | refine_cv_staged_k4 | 两阶段 K=4 | `refine_cv_staged.py` | record 目录 | 55.5% |
| 18 | refine | **REFINE 基线（无 CEV-GR）** | `refine_clean.py` | `percase_refine.json` | 55.0% |
| 18a | sc_cover | 8维框架（无CEV-GR，SC summarizer管线） | `sc_cover.py` | `percase_sc_cover.json` | 56.5% |
| 19 | refine_cv_discrim | 方案C1：判别优先 | `refine_cv_discrim.py` | `percase_refine_cv_discrim.json` | 55.0% |
| 20 | refine_cv_staged | 两阶段 K=6 | `refine_cv_staged.py` | record 目录 | 54.5% |
| 21 | refine_cv_staged_k8 | 两阶段 K=8 | `refine_cv_staged.py` | record 目录 | 54.5% |
| 22 | refine_cv_p0 | P0：主动 top-2 | `refine_cv_p01.py` | `percase_refine_cv_p0.json` | 54.0% |
| 22a | roleplay_cover_abl | 消融：roleplay底座 + 8维框架（无CEV-GR） | `roleplay_cover_abl.py` | `percase_roleplay_cover_abl.json` | 53.5% |
| 23 | react | ReAct | `react.py` | `percase_react.json` | 53.0% |
| 24 | refine_cv_p0p1 | P0+P1 联合 | `refine_cv_p01.py` | `percase_refine_cv_p0p1.json` | 52.0% |
| 25 | roleplay | roleplay（交互基线） | `roleplay.py` | `percase_roleplay.json` | 49.0% |
| 25a | react_cover | 方案A移植：react底座 + 8维框架 + CEV-GR | `refine_cv_cover_ports.py` | `percase_react_cover.json` | 50.0% |
| 26 | react_cv | ReAct 底座 + 四检查 | `refine_cv_ports.py` | `percase_react_cv.json` | 46.5% |
| 27 | react_cv2 | ReAct + 专用 CEV-GR | `react_cv2.py` | `percase_react_cv2.json` | 45.0% |
| 28 | mediq_minq | MEDIQ（强制问4轮） | `mediq_abstention.py` | `percase_mediq_minq.json` | 44.7% |
| 29 | mediq_abstention | MEDIQ（abstention） | `mediq_abstention.py` | `percase_mediq_abstention.json` | 44.0% |
| 30 | medclarify_minq | MedClarify（强制问4轮） | `medclarify.py` | `percase_medclarify_minq.json` | 31.2% |
| 31 | medclarify | MedClarify（DEIG） | `medclarify.py` | `percase_medclarify.json` | 28.0% |
| 32 | uot | UoT（信息增益） | `uot.py` | `percase_uot.json` | 26.0% |

> staged（#17/20/21）准确率来自 `docs/two-stage-explore-exploit-results.md`，无 percase 文件；其余来自 percase `correct` 字段。

---

## 2. 分组明细

### 2.1 论文 5 方法（`PAPER_MODES`）

| mode | 说明 | 场景代码 | 准确率 |
|---|---|---|---|
| cot | 静态全信息上界 | `cot.py` | 94.0% |
| sc | 总结诊断 | `sc.py` | 59.0% |
| refine | REFINE 基线（无 CEV-GR） | `refine_clean.py` | 55.0% |
| react | ReAct | `react.py` | 53.0% |
| roleplay | 交互基线 | `roleplay.py` | 49.0% |

### 2.2 CEV-GR 系列（`refine_cv*`）

| mode | 说明 | 场景代码 | 准确率 |
|---|---|---|---|
| refine_cv | 跨模型 DeepSeek 四检查 | `refine_cv.py` | 58.5% |
| refine_cv_qwen | 同模型四检查 | `refine_cv_qwen.py` | 57.5% |
| refine_cv_no_decisive | 跨模型三检查 | `refine_cv_ablations.py` | 57.5% |
| refine_cv_qwen_no_decisive | 同模型三检查 | `refine_cv_ablations.py` | 60.0% |
| refine_cv_qwen_check | 只 claim | `refine_cv_ablations.py` | 59.8% |
| refine_cv_qwen_gap | 只 gap | `refine_cv_ablations.py` | 59.0% |
| refine_cv_qwen_gap_refute | gap+refute | `refine_cv_ablations.py` | 58.5% |

### 2.3 采集层方案（主战场，含负结果）

| mode | 说明 | 场景代码 | 准确率 | vs refine_cv |
|---|---|---|---|---|
| **refine_cv_cover** | **方案A：去效率 + 8维框架** | `refine_cv_cover.py` | **65.0%** | **+6.5pp** ✅ |
| refine_cv_cover_plan | 方案B：采集规划器 | `refine_cv_cover_plan.py` | 59.8% | +1.3pp ❌ |
| refine_cv_p1 | P1：base-rate prior | `refine_cv_p01.py` | 59.0% | +0.5pp ❌ |
| refine_cv_p0 | P0：主动 top-2 | `refine_cv_p01.py` | 54.0% | −4.5pp ❌ |
| refine_cv_p0p1 | P0+P1 | `refine_cv_p01.py` | 52.0% | −6.5pp ❌ |
| refine_cv_staged(_k4/_k8) | 两阶段探索-区分 | `refine_cv_staged.py` | 54.5–55.5% | −3~−4pp ❌ |
| refine_cv_discrim | 方案C1：判别优先 | `refine_cv_discrim.py` | 55.0% | −3.5pp ❌ |

### 2.4 消融对照（方案A 的因果拆解）

| mode | 说明 | 场景代码 | 准确率 | 回答的问题 |
|---|---|---|---|---|
| refine | 基线（无 CEV-GR） | `refine_clean.py` | 55.0% | 起点 |
| refine_cv | + CEV-GR 四门控 | `refine_cv.py` | 58.5% | CEV-GR 贡献 = +3.5pp |
| refine_cv_freebudget | CEV-GR + 去效率 | `refine_cv_freebudget.py` | 57.5% | 去效率单独贡献 = −1.0pp |
| refine_cv_cover | CEV-GR + 去效率 + 8维框架 | `refine_cv_cover.py` | 65.0% | 8维框架（在CEV-GR底座）= +7.5pp |
| **refine_cover** | **去效率 + 8维框架（无 CEV-GR）** | `refine_cover.py` | **60.0%** | **8维框架独立 +5.0pp，非协同依赖** |

### 2.5 诊断层去偏置

| mode | 说明 | 场景代码 | 准确率 |
|---|---|---|---|
| refine_cv_debias | 基线 + 分级强制 | `refine_cv_debias.py` | 59.8% |
| refine_cv_cover_debias | 方案A + 分级强制 | `refine_cv_debias.py` | 58.0% |

### 2.6 外部论文方法移植（零训练 baseline）

| mode | 说明 | 场景代码 | 准确率 |
|---|---|---|---|
| mediq_minq | MEDIQ 强制问4轮 | `mediq_abstention.py` | 44.7% |
| mediq_abstention | MEDIQ abstention | `mediq_abstention.py` | 44.0% |
| medclarify_minq | MedClarify 强制问4轮 | `medclarify.py` | 31.2% |
| medclarify | MedClarify DEIG | `medclarify.py` | 28.0% |
| uot | UoT 信息增益 | `uot.py` | 26.0% |

### 2.7 跨底座移植（CEV-GR 通用性 + 方案A完整插件）

**完整插件（8维框架 + CEV-GR）跨底座对照**（本次补齐）：

| 底座 | 原样 | +CEV-GR | +8维框架(无CEV-GR) | 完整插件(8维+CEV-GR) | 判定 |
|---|---|---|---|---|---|
| refine | 55.0% | 58.5% (refine_cv) | 60.0% (refine_cover) | **65.0%** (refine_cv_cover) | ✅ +10.0 |
| roleplay | 49.0% | 57.5% (roleplay_cv) | 53.5% (roleplay_cover_abl) | **64.0%** (roleplay_cover) | ✅ **+15.0** |
| sc | 59.0% | 58.5% (==refine_cv) | **56.5%** (sc_cover) | 65.0% (==refine_cv_cover) | ⚠️ 8维框架−2.5 |
| react | 53.0% | 46.5% (react_cv) | — | **50.0%** (react_cover) | ❌ −3.0 |

**三个新发现（本次移植实验的核心产出）**：

1. **roleplay 底座大胜（64.0%，+15.0pp）**：完整插件在 roleplay 上从 49% 拉到 64%，**几乎追平 refine 底座（65.0%）**。8维框架独立贡献 **+4.5pp**（49.0→53.5，`roleplay_cover_abl` 补跑），与 refine 底座上的 +5.0pp 一致；两组件单独贡献之和（+4.5+8.5=+13.0）< 完整插件（+15.0），存在约 +2.0pp 正协同。→ **8维框架 + CEV-GR 是底座无关的、可加的两组件插件**，泛化性坐实。

2. **sc 底座暴露 8维框架的诊断管线依赖（56.5%，−2.5pp）**：sc_cover（8维框架 + SC summarizer 一步诊断管线，无 CEV-GR）比 sc 原样 59.0% **还低 2.5pp**，而 refine_cover（同样的 8维框架，但在 refine 验证环管线上）比 refine 原样 +5.0pp。→ **8维框架的采集增益需要"验证环（gap-driven 补问）"来消化广覆盖证据**；在 SC 的一步式 summarizer+diagnostician 管线上，广覆盖反而稀释了 summary，略有害。

3. **react 底座结构性不兼容再次确认（50.0%，−3.0pp）**：react_cover（8维框架+CEV-GR）比 react 原样 53.0% 仍低 3.0pp，但比 react_cv（46.5%）回升 3.5pp。→ 8维框架部分缓解了 CEV-GR 在 react 上的 give_up 损伤，但 react 的"采集-诊断耦合 + [DIAGNOSIS]=终点"结构性冲突仍主导，无法逆转。

| mode | 说明 | 场景代码 | 准确率 |
|---|---|---|---|
| roleplay_cover | roleplay + 8维框架 + 四检查 | `refine_cv_cover_ports.py` | **64.0%** |
| roleplay_cover_abl | roleplay + 8维框架（无CEV-GR） | `roleplay_cover_abl.py` | 53.5% |
| sc_cover | sc + 8维框架（无CEV-GR，summarizer管线） | `sc_cover.py` | 56.5% |
| react_cover | ReAct + 8维框架 + 四检查 | `refine_cv_cover_ports.py` | 50.0% |
| roleplay_cv | roleplay + 四检查 | `refine_cv_ports.py` | 57.5% |
| react_cv | ReAct + 四检查 | `refine_cv_ports.py` | 46.5% |
| react_cv2 | ReAct + 专用 CEV-GR | `react_cv2.py` | 45.0% |

---

## 3. 关键消融链（因果对照，供复盘）

```
refine (55.0%)  ──+CEV-GR四门控(+3.5)──►  refine_cv (58.5%)
                                              │
                                              ├─ +去效率(−1.0) ─► freebudget (57.5%)
                                              │
                                              └─ +去效率+8维框架(+6.5) ─► cover/方案A (65.0%)

refine (55.0%)  ──+去效率+8维框架(+5.0)──►  refine_cover (60.0%)   ← 本次消融
refine_cover (60.0%)  ──+CEV-GR四门控(+5.0)──►  cover/方案A (65.0%)
```

**核心问题已有答案**：方案A 的 +10.0pp（refine→cover）是**两个独立机制的近似可加贡献**，而非协同依赖：

| 机制 | 独立贡献 | 测量方式 |
|---|---|---|
| 8 维证据框架 | **+5.0pp** | refine(55.0) → refine_cover(60.0)，无 CEV-GR 底座 |
| CEV-GR 四门控 | **+5.0pp** | refine_cover(60.0) → refine_cv_cover(65.0)，在 cover 底座上 |

- 8 维框架离开 CEV-GR 仍值 +5.0pp（不是贴着 refine 的协同依赖）→ **框架有独立贡献**。
- CEV-GR 在 cover 底座上值 +5.0pp，比它在 refine 底座上的 +3.5pp 还略高 → **两者叠加有轻微正协同**（框架采来的广覆盖证据，被门控消化得更充分）。
- 总账：5.0 + 5.0 = 10.0pp，正好等于 refine→cover 的总增益（+10.0pp）。

**roleplay 底座同样闭合**（`roleplay_cover_abl` 补跑，53.5%）：

```
roleplay (49.0%) ──+8维框架(+4.5)──► roleplay_cover_abl (53.5%)
                                        └─+CEV-GR(+10.5)──► roleplay_cover (64.0%)
roleplay (49.0%) ──+CEV-GR(+8.5)──► roleplay_cv (57.5%)
```

- 8维框架在 roleplay 底座上的独立贡献 **+4.5pp**，与 refine 底座 +5.0pp 一致 → 采集组件底座无关。
- 两组件单独贡献之和 +13.0pp < 完整插件 +15.0pp → 存在约 **+2.0pp 正协同**（弱基线补短板时协同更明显）。

---

## 4. 统一负结果规律（复盘的锚）

采集层凡让模型「排序/选择/优先级」的方案全部 ≤ 基线：P0(−4.5) / P0P1(−6.5) / staged(−3~−4) / 方案C1(−3.5) / 方案B(仅+1.3)。**唯一正结果是「静态清单」范式的方案A(+6.5)**——给医生一份"覆盖什么"的清单，而非让它"选什么"。

**本次消融（refine_cover）补上最后一环**：8 维框架的贡献是独立的（无 CEV-GR 底座上仍 +5.0pp），CEV-GR 四门控是第二独立贡献（cover 底座上 +5.0pp）。方案A = 8维框架(+5.0) + CEV-GR(+5.0) ≈ +10.0pp 的可加组合，且两者叠加有轻微正协同。这坐实了"静态清单范式有效"的因果归属——不是框架抱了门控的大腿，而是两个机制各自独立、又互补。
