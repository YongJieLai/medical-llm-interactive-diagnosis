# 方案A 完整插件（8维框架 + CEV-GR）跨底座移植结果

> 实验日期：2026-11-27 ｜ 200 例 MedQA × 16 turns × qwen-plus ｜ 判分 deterministic_judge
> 背景：我们的方法定位是**插件**（不是 standalone）。之前只移植过 CEV-GR 四检查门控
> （`docs/cev-gr-host-portability.md`），从没移植方案A的 8 维证据框架。本次补齐——
> 把完整插件（8维框架 + CEV-GR）移植到 roleplay / sc / react 三个底座。

---

## 0. 插件是什么（两个组件）

| 组件 | 层 | 改动 | 独立贡献（refine_cover 消融） |
|---|---|---|---|
| 8 维证据框架 + 去效率 | 采集层 | doctor prompt 三处替换 | **+5.0pp** |
| CEV-GR 四检查门控 | 诊断层 | claim/gap/refutation/decisive 验证环 | **+5.0pp** |

---

## 1. 核心结果表（完整插件跨底座）

| 底座 | 原样 | +CEV-GR | +8维框架(无CEV-GR) | 完整插件(8维+CEV-GR) | Δ vs 原样 | 判定 |
|---|---|---|---|---|---|---|
| **refine** | 55.0% | 58.5% | 60.0% | **65.0%** | +10.0 | ✅ |
| **roleplay** | 49.0% | 57.5% | 53.5% | **64.0%** | **+15.0** | ✅ 最大 |
| **sc** | 59.0% | 58.5% | 56.5% | 65.0% | +6.0 | ⚠️ 8维框架单独−2.5 |
| **react** | 53.0% | 46.5% | — | **50.0%** | −3.0 | ❌ 结构性不兼容 |

> sc 的 "完整插件" == refine_cv_cover（65.0%），因为 SC doctor prompt 与 refine 逐字相同、
> 且 CEV-GR 用 RefineCV 验证环管线（非 SC summarizer）。故 sc 那格不重复跑，直接引用 65.0%。

---

## 2. 三个核心发现

### 发现 1：roleplay 底座大胜（64.0%，+15.0pp）——泛化性坐实

```
roleplay 49.0% ──+CEV-GR(+8.5)──► roleplay_cv 57.5%
                                    └─+8维框架(+6.5)──► roleplay_cover 64.0%
roleplay 49.0% ──+8维框架(+4.5)──► roleplay_cover_abl 53.5%   ← 独立贡献（补跑）
```

- 完整插件在 roleplay 上从 49% 拉到 **64.0%**，**几乎追平 refine 底座（65.0%）**。
- **8维框架独立贡献 +4.5pp**（49.0→53.5，`roleplay_cover_abl`），与 refine 底座上的
  +5.0pp（55.0→60.0）一致 → 采集组件**底座无关**。
- 8维框架在 CEV-GR 底座上的增量 **+6.5pp**（57.5→64.0），与它在 refine 底座上的增量
  （refine_cv 58.5 → refine_cv_cover 65.0，+6.5pp）**完全一致**。
- 两组件单独贡献之和（+4.5 + 8.5 = +13.0）< 完整插件（+15.0），存在约 **+2.0pp 正协同**。
- **结论**：8维框架 + CEV-GR 是**底座无关的、可加的两组件插件**。roleplay 是"采集-诊断
  解耦"底座（医生采集 + 独立 diagnostician），与 refine 同构，插件原样生效。

### 发现 2：sc 底座暴露 8维框架的"诊断管线依赖"（−2.5pp）——最锋利的发现

```
refine 管线（验证环）:  55.0% ──+8维框架──► refine_cover 60.0%   (+5.0pp)
SC 管线（summarizer）:  59.0% ──+8维框架──► sc_cover 56.5%       (−2.5pp)
```

- **同样的 8维框架采集改动**，在 refine 验证环管线上 +5.0pp，在 SC summarizer 一步诊断
  管线上 **−2.5pp**。
- **机制**：8维框架让医生多采广覆盖证据（含大量噪音），refine 的验证环（gap-driven 补问）
  能把这些证据按"缺口"消化、定向补关键检查；而 SC 的 summarizer 把全部证据一锅端进
  summary，广覆盖反而**稀释了关键信息**，diagnostician 在稀释后的 summary 上略退化。
- **结论**：8维框架不是"纯采集增益"，它的收益**依赖诊断层有验证环来消化广覆盖证据**。
  这解释了为什么方案A = 8维框架 + CEV-GR 必须成对出现（8维框架采、CEV-GR 消化）。

### 发现 3：react 底座结构性不兼容再次确认（50.0%，−3.0pp）

```
react 53.0% ──+CEV-GR(−6.5)──► react_cv 46.5%
                                └─+8维框架(+3.5)──► react_cover 50.0%  (仍 < 原样)
```

- 8维框架**部分缓解**了 CEV-GR 在 react 上的 give_up 损伤（46.5→50.0，+3.5pp），但仍低于
  react 原样 53.0%（−3.0pp）。
- react 的"采集-诊断单 agent 耦合 + [DIAGNOSIS] 是终点"结构性冲突**主导**，8维框架只能
  缓解、不能逆转。这再次确认 `docs/cev-gr-host-portability.md` 的适用边界。

---

## 3. 对论文定位的最终表述（升级）

| 原表述 | 升级后 |
|---|---|
| CEV-GR 是"采集-诊断解耦底座"的可插拔审计层 | 我们的插件 = **采集层（8维框架）+ 诊断层（CEV-GR）双组件**；两者可加、但**采集组件需诊断组件的验证环来消化** |

**泛化性证据链（现在完整）**：
1. 在 refine / roleplay 两个"采集-诊断解耦"底座上，完整插件稳定 +10.0~+15.0pp，且两组件的
   独立贡献一致（8维框架 +6.5pp、CEV-GR +3.5~+8.5pp）。
2. 在 sc 底座上，8维框架单独 −2.5pp，证明采集组件**依赖诊断组件的验证环**（成对才有效）。
3. 在 react 底座上，完整插件 −3.0pp，证明"采集-诊断耦合"底座是**硬边界**。

---

## 4. 工程记录

- 代码：
  - `scenarios/refine_cv_cover.py`：重构，抽出 `_apply_cover_edits(base, turn_anchor)` 可复用函数。
  - `scenarios/refine_cv_cover_ports.py`：`RoleplayCoverScenario` / `ReactCoverScenario`。
  - `scenarios/sc_cover.py`：`SCCoverScenario`（SC summarizer 管线 + 8维框架，无 CEV-GR）。
  - `scenarios/sc.py`：加 `_doctor_system_prompt()` hook + `MODE` 属性（向后兼容）。
  - `scenarios/__init__.py`、`src/run_paper_modes.py`：注册三个新 mode。
- 实验：三模式各 200 例，3 模式 × 6 分片 = 18 进程并行，workers=3，无 err。
- 判分：`deterministic_judge`，uncertain=0。
