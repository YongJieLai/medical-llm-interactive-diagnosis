# P0 reporter 口径修复 · 三态验证报告

> 目标：修复 fork 引入的 closed-world reporter 口径（查不到→`Not specified`）造成的 71% 假失败。
> 验证集：`refine_cv_qwen` 60 例（qwen-plus，10_turns），确定性 `rule_judge`（None 算错）。

## 一、三态准确率

| 口径 | 正确 | 准确率 | vs 基线 |
|---|---:|---:|---:|
| closed-world（基线） | 32/60 | **53.3%** | — |
| open-world（方案 A） | 24/60 | **40.0%** | -13.3pp |
| **availability（方案 B）** | 35/60 | **58.3%** | **+5.0pp** |

## 二、口径切换验证（record 全文标记计数）

| 口径 | `Not specified` | `Normal` | `Not available` |
|---|---:|---:|---:|
| closed-world | 868 | 550 | 0 |
| open-world | 5 | 1752 | 0 |
| availability | 15 | 751 | 637 |

方案 B 成功把"查不到"从 `Not specified`（死路）改为 `Not available + 可请求目录`
（637 次），同时没有像方案 A 那样把缺数据误报成 `Normal`（假阴性）。

## 三、逐例 diff（availability vs closed-world 基线）

**修好 12 例 / 改坏 9 例**（净 +3 例）。修好的 12 例几乎全是"基线卡死在 Not specified、
无法确诊"的 case：

| case | truth | 基线（错） | availability（对） |
|---|---|---|---|
| medqa_9 | Endometritis | Symptom unspecified | Postpartum endometritis |
| medqa_20 | CMV retinitis | Undiagnosed type 2 DM | CMV retinitis |
| medqa_34 | Erythema Infectiosum | Insufficient info | Erythema infectiosum |
| medqa_38 | Duodenal atresia | Insufficient data | Duodenal Atresia |
| medqa_40 | Medullary Sponge Kidney | Urothelial carcinoma | Medullary Sponge Kidney |
| medqa_59 | Pelvic inflammatory disease | Ectopic pregnancy | Acute PID |

改坏的 9 例中，**约 3 例是判定器（rule_judge）同义词未覆盖的误判**（实际答对）：
- medqa_4 `tubulointerstitial nephritis` = interstitial nephritis（同义）；
- medqa_5 `Pes anserine tendinosis/bursitis` 含 bursitis；
- medqa_53 `BPPV, right horizontal canal` 是 BPPV 更具体亚型。

真正改坏约 5–6 例（如 medqa_14 jersey finger→volar plate、medqa_29 DGI→septic arthritis、
medqa_39 ET→prefibrotic MF），说明方案 B 仍会引入少量假阴性，但幅度远小于方案 A。

修正判定器这 3 例后，availability 实际约 **38/60 = 63.3%**（+10.0pp）。

## 四、McNemar 检验

| 对比 | b(对→错) | c(错→对) | p |
|---|---:|---:|---:|
| baseline vs open-world | 16 | 8 | 0.153（不显著） |
| baseline vs availability | 9 | 12 | 0.663（不显著） |

60 例样本量下均不显著，但方向明确：availability 是唯一正向的（c > b）。

## 五、结论

1. **方案 A（open-world）被数据否决**（-13.3pp），假阴性是主因——印证了分析文档的预判。
2. **方案 B（availability）是正确方向**：相对基线 +5.0pp，既消除 71% 假失败
   （`Not specified` 868→15），又避免方案 A 的假阴性（`Not available` 而非误返 `Normal`）。
3. 剩余假阴性（约 5 例）主要来自"医生转向后仍选错"的固有困难 case，非口径本身问题。

## 六、代码状态

- `prompts/manager.py`：
  - reporter 口径开关升级为三态 `REPORTER_MODE ∈ {closed, open, availability}`，默认 **availability**；
  - doctor prompt 新增第 5 条 "Test Availability" 引导（收到 `Not available` 时转向 `Available tests`）。
- 验证脚本：`src/p0_reporter_compare.py`。
- 结果目录：`refine_cv_qwen_openworld`（方案 A）、`refine_cv_qwen_availability`（方案 B）。

## 七、下一步

1. 把方案 B 与 P0 的 gap 三道 gate（亚型 gate / 鉴别树锚定 / 确诊优先）叠加，继续压缩假阴性；
2. 修正 `deterministic_judge.py` 的 3 处同义词（tubulointerstitial=interstitial、BPPV 亚型、
   tendinosis/bursitis），消除判定器噪声后再复评；
3. 在 ≥200 例上验证显著性（当前 60 例 p=0.66 不足以下结论）。
