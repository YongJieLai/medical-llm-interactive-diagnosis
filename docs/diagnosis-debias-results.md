# 诊断层去偏置（方案 1：诊断分级强制）——负结果

> 状态：**已完成，负结果**（2026-11-27）
> 背景：`error-distribution-and-improvement.md`（overdiagnosis 占错题 14–16%）、`p0p1-improvement-results.md`（P1 base-rate prior 负结果）。
> 本文回答：诊断层去偏置的第 1 条路——「诊断分级强制」是否有效，以及它揭示了什么。

## 0. 结论速览（30 秒）

1. **分级强制在基线底座上持平**（59.8% vs 58.5%，净 −2 例，p=0.87）：基线瓶颈在采集（覆盖 31%），分级强制治错了病。
2. **分级强制在方案 A 底座上大幅掉点**（65.0% → 58.0%，净 −14 例，p=0.071）：在证据已充足的 cover 底座上，分级约束制造了新的诊断漂移。
3. **overdiagnosis 没被压掉，反而转移成 underdiagnosis**：cover 底座下 overdiagnosis 6→9 例（8.6%→10.7%，不降反升）、underdiagnosis 2→9 例（+7 例暴增）、judge_false_negative 1→5 例。
4. **关键洞察：方案 A 本身已经把 overdiagnosis 从 14% 压到 8.6%**——诊断层去偏置的独立空间很小，且 prompt 层的"倾向/层级"直接干预已被 P1 + 方案 1 双重否定。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 数据 | MedQA 200 例 / 16 turns / qwen-plus，空起步 |
| 判分 | `deterministic_judge`（规则层 + DeepSeek 兜底），无缓存污染 |
| 覆盖率 | `analyze_coverage.py`，关键原子事实（LLM 标注）采集率 |
| 错误分类 | DeepSeek 逐例根因标注（off_system / wrong_entity / overdiagnosis / underdiagnosis / give_up / judge_false_negative） |

方案 1 的改动（`refine_cv_debias.py`）：在 diagnostician 的 `[DIAGNOSIS]` 段前插入 `[ESCALATION]`
段（列出"想升级但证据不足"的候选，明确不计入答案），并让 `[DIAGNOSIS]` 段只给"证据能直接
确立的最高诊断层级"。`[DIAGNOSIS]` 保持在最后，`extract_action` 提取的最终答案天然不含升级候选。
纯 prompt 改动、零额外调用、零训练。

两个变体：
- `refine_cv_debias`：基线 + 分级强制（对照基线、P1）
- `refine_cv_cover_debias`：方案 A（广覆盖采集）+ 分级强制（对照方案 A）

---

## 2. 结果总览（五模式）

| 模式 | 准确率 | 覆盖率中位 | 正确组覆盖 | 错误组覆盖 | 轮次中位 |
|---|---|---|---|---|---|
| refine_cv（基线） | 58.5% (117/200) | 31.0% | 33.3% | 30.0% | 15 |
| refine_cv_p1（P1 先验） | 59.0% (118/200) | 33.3% | 35.7% | 28.9% | 15 |
| **refine_cv_debias（分级强制）** | **59.8% (119/199)** | 31.8% | 33.3% | 29.7% | 15 |
| refine_cv_cover（方案 A） | **65.0% (130/200)** | 50.0% | 54.3% | 40.0% | 16 |
| **refine_cv_cover_debias（A+分级强制）** | **58.0% (116/200)** | 51.9% | 54.5% | 42.3% | 16 |

**配对净效应（McNemar）**：

| 对比 | n | 改坏 | 改对 | 净 | p |
|---|---|---|---|---|---|
| 分级强制 vs 基线 | 199 | 19 | 17 | **−2** | 0.868 |
| P1 先验 vs 基线 | 200 | 21 | 20 | −1 | 1.000 |
| A+分级强制 vs 方案 A | 200 | 33 | 19 | **−14** | 0.071 |

覆盖率中位在 cover_debias 上还略升（50.0%→51.9%），证明掉点**完全来自诊断层**，与采集无关。

---

## 3. 错误分类（cover vs cover_debias，决定性证据）

| 错误类型 | 方案 A（70 错） | A+分级强制（84 错） | 变化 |
|---|---|---|---|
| off_system | 18 (25.7%) | 22 (26.2%) | +4 |
| wrong_entity | 41 (58.6%) | 36 (42.9%) | −5 |
| **overdiagnosis** | **6 (8.6%)** | **9 (10.7%)** | **+3（不降反升）** |
| **underdiagnosis** | **2 (2.9%)** | **9 (10.7%)** | **+7（暴增）** |
| give_up | 2 (2.9%) | 3 (3.6%) | +1 |
| judge_false_negative | 1 (1.4%) | 5 (6.0%) | +4 |

**三个信号**：

1. **overdiagnosis 没被压掉**（6→9 例）。分级强制本意是"升级需举证"，但模型把 overdiagnosis
   的部分换成了 underdiagnosis，overdiagnosis 本身不降反升。
2. **underdiagnosis 从 2 例暴增到 9 例**。这正是分级强制的"保守降级"副作用：把具体病降级成
   宽泛/低锐度标签。逐例可见：Legg-Calvé-Perthes → Transient synovitis、Ewing sarcoma →
   "小圆细胞瘤（未分类）"、Tetralogy of Fallot → "Unspecified irritability"。
3. **judge_false_negative 从 1 例升到 5 例**。分级强制让输出更啰嗦/带修饰，判分器误杀增多
   （如 "Contact lens–associated microbial keratitis" 被判错，实质接近 truth "Pseudomonas keratitis"）。

---

## 4. 根因：prompt 层"倾向/层级"干预注定失败（P1 + 方案 1 双重负证据）

P1（base-rate prior）和方案 1（分级强制）看似相反——P1 拉向"常见"，方案 1 拉向"证据能确立的
层级"——但**失败根因完全相同**：

> LLM 的诊断输出是**单点的、锚定在其训练分布上**的。任何"往哪个方向偏/怎么分级"的指令，
> 都会与它本来的（常常是正确的）判断打架，结果只是把一种偏置**转移**成另一种，而非消除：
> - P1 把 overdiagnosis（12→3 例）转移成 wrong_entity（29→35 例）；
> - 方案 1 把 overdiagnosis 的部分转移成 underdiagnosis（2→9 例）+ 输出啰嗦（judge_false_negative 1→5 例）。
>
> 在**证据充足**时这种干预尤其有害：cover 底座下诊断器本来靠充足证据能答对，分级约束反而
> 制造了"该不该升级"的犹豫，把 33 例原本答对的 case 改错。

**这与"采集层"的教训是同一枚硬币的两面**：方案 A 成功恰恰因为它**不干预模型倾向**，只给
静态的"该覆盖哪些证据类别"框架，把"怎么诊断"留给模型自己的证据纪律；而 P1 / 方案 1 / 方案 B
都失败在**直接干预模型的判断倾向**（假设倾向 or 诊断倾向）。

---

## 5. 关键洞察：诊断层去偏置不是当前主战场

方案 A 之后，overdiagnosis 其实已经**大幅下降**：

| | 基线 refine_cv | 方案 A cover |
|---|---|---|
| overdiagnosis | 12 例（14%） | **6 例（8.6%）** |

原因：overdiagnosis 的一个重要来源是"证据不足时靠猜测升级"，而方案 A 把覆盖率从 31% 抬到
50%，诊断器有据可依，就不再乱升级。**采集层（方案 A）顺带解决了大部分诊断层偏置。**

因此，当前剩余的错误结构是：

- cover 底座 70 例错题里，**off_system（26%）+ wrong_entity（59%）合计 85%**，仍是"采集/
  实体匹配"问题；
- 诊断层 overdiagnosis 只剩 **6 例（8.6%）**，且 prompt 干预（P1、方案 1）已证明无效。

**诊断层去偏置的独立空间很小（≤6 例），且可用的 prompt 手段已被双重否定。真正的主战场
依然是采集层——特别是 wrong_entity（59%，41 例）这个尚未深挖的大头。**

---

## 6. 下一步

1. **[优先] 深挖 wrong_entity（cover 底座 41 例，59%）**：区分"同系统真错实体" vs "实体粒度
   不同（判分口径）"。若是后者（如 Pseudomonas keratitis vs microbial keratitis），说明真实
   准确率被低估，可据此校准判分；若是前者，则是"同一系统内还缺哪条判别证据"的采集问题。
2. **[可选] 方案 B′（category-only 采集规划器）**：规划器只输出 `{category, why}`、不输出
   question，干净消融"缺口识别价值 vs 问题生成害处"。这是采集层唯一未试的方向。
3. **[不做] 诊断层 prompt 干预**：P1 + 方案 1 双重负证据已关闭这条路；若坚持治 overdiagnosis，
   应改走**确定性规则门控**（诊断严重/罕见病时强制检查 summary 有无对应阳性证据），但空间仅 6 例。
