# GAP CHECK 机制深度分析

> 数据：`refine_cv_qwen` 完整版 60 例 trace（`benchmarks/eid-benchmark/results/medqa/refine_cv_qwen/qwen-plus/10_turns/record/`）
> 对象：CEV-GR 三组件中的 **gap check**（REFINE 唯一原生继承的组件）

---

## 一、gap check 是怎么设定的

### REFINE 原文（arXiv 2601.19773 附录 E）
Verifier 做 **Data Sufficiency 二元门控**：
```
[DECISION] PASS（证据充分） / INCOMPLETE（证据不足，诊断过早）
[FEEDBACK] INCOMPLETE 时：指出还缺哪个具体信息
```
逻辑单向：**只问"够不够"，不问"对不对、有没有更好的"**。

### 我们的实现（refine_cv.py + verifier prompt）
三步走：
1. **diagnostician** 自报 `[GAPS]`（生成端自我认知缺口）
2. **verifier**（另一模型）做 GAP CHECK：列 top 2–3 个"能区分当前诊断 vs 主要替代诊断"的缺失信息，输出结构化 `{"info", "discriminates"}`
3. **规则树**：`if gaps and turn < max_turns-1: 取 gaps[0] → 让 doctor 追问一轮 → GATHER`

与原文的关键差异：我们把"充分性门控"改成了"鉴别诊断缺口枚举"，并且**只要 gaps 非空就追问，没有 PASS/INCOMPLETE 的明确门控**。

---

## 二、实证发现的问题（60 例 trace）

### 总体统计
| 指标 | 数值 |
|---|---|
| 触发过 gap 的 case | **54 / 60**（90%）|
| gap 触发总轮次 | 103 |
| gap 条目总数 | 309 |
| 每轮 gap 数量 | **恒为 3**（103 轮全是 3 个）|
| 同一 case gap 触发 ≥2 轮（死循环）| **36 例** |
| 触发 gap 的 case 最终答对率 | 53.7%（29/54）|
| 未触发 gap 的 case 最终答对率 | 83.3%（5/6）|

### 问题 1：gap 混淆「诊断」与「分级/分期/治疗决策」（最普遍）
大量 gap 追问的**不是"这是不是 X 病"，而是"X 病到了第几期/分级"**，诊断已对时仍在空转：

| case | 诊断（已对）| gap 在问什么 |
|---|---|---|
| 22 Malignant melanoma | ✅ | Breslow thickness、ulceration、margin、mitotic rate（**T 分期**）|
| 8 Phyllodes tumor | ✅ | stromal mitotic count、tumor size（**分级 + 手术决策**）|
| 24 Meigs' syndrome | ✅ | 反复问"病理确诊卵巢肿块"（**reporter 根本没有病理数据**）|

→ 三轮 gap 追问不改变诊断，纯浪费轮次（36 例死循环的直接来源）。

### 问题 2：追问「环境中不存在的数据」→ 必然死循环
verifier 反复要求"病理确诊""核型分析""超声心动图"等 **reporter（化验员）无法返回的信息**，问满 3 轮也问不到。

### 问题 3：每轮恒 3 个 gap —— 凑数式输出
prompt 说"top 2–3"，但 verifier 把"列 gap"当格式义务，**103 轮全部凑满 3 个**。大量 gap 是"Patient age""White blood cell count"这类已在 summary 里的基础信息，属于**无病呻吟**。

### 问题 4：gap 追问导致诊断「退化 / 放弃」（直接破坏）
| case | 诊断轨迹 | 真相 |
|---|---|---|
| **38** | Duodenal atresia（对）→ **Insufficient data**（放弃）| Duodenal atresia |
| **9** | Anxiety disorder → **Symptom unspecified**（泛化）| Endometritis |
| **34** | Measles → **Insufficient info**（不确定）| Erythema Infectiosum |

gap 让 verifier 反复宣称"信息不足"，把原本正确的诊断逼成"无法诊断"。

### 问题 5（根因）：gap check 受限于 summarizer 的完整性
**case 38 的讽刺实锤**：
- diagnostician 第 1 轮就正确诊断 Duodenal atresia（依据"双泡征""胆汁性呕吐"）
- verifier 却说"缺 double bubble sign / bilious vomiting"——**这些证据明明在对话里**
- 真相是 **summarizer 在整理时丢了这些关键证据**，verifier 基于残缺的 summary 误判"缺 gap"
- 最终把正确诊断逼成"放弃"

→ **gap check 说"缺 X"，可能是真的缺，也可能是 summarizer 把 X 丢了**。二者当前无法区分。

### 问题 6：规则树只取 gaps[0]，不评估可操作性/价值
`top = gaps[0]`，而第一个 gap 往往是"病理确诊"这类死胡同，后面真正能区分诊断的 gap 被忽略。

---

## 三、改进方案（按优先级）

| # | 改进 | 对应问题 | 落地方式 |
|---|---|---|---|
| **P0-1** | gap 加「诊断相关性」过滤：`discriminates` 必须是"当前诊断 vs 另一候选诊断"，过滤"当前诊断内部的分级/分期/治疗" | 1、3 | 规则：若 gap 目标仍是当前诊断（只是更深分级），跳过 |
| **P0-2** | gap 加「可获取性」检查：只追问病史/体格/常规化验类，跳过病理确诊/基因检测/核型等 reporter 无数据项 | 2 | 关键词黑名单 + 匹配 exam_facts 可达项 |
| **P0-3** | 限制 gap 追问轮数（同一 case 最多 2 轮） | 2、死循环 | 规则树加计数 |
| **P1-1** | gap 追问后保护「诊断稳定性」：若从明确诊断退化为"无法诊断/信息不足"，回退到退化前的明确诊断 | 4 | 规则树记录上一轮明确诊断，退化为 None 时回退 |
| **P1-2** | 修复 gap 输入：让 verifier 同时看 summary + 原始对话（或关键证据不丢失的摘要），消除 summarizer 丢信息导致的误判 | 5 | 改 `_verify` 的 user prompt 传入原始对话摘录 |
| **P1-3** | verifier prompt 收紧：明确"只报能改变诊断结论的缺口，诊断已确定就返回空 list；不要报告分期/分级/治疗信息" | 3 | 改 `get_verifier_cv_system_prompt` |

---

## 四、结论

gap check 是 REFINE 唯一原生继承的组件，但我们的"鉴别诊断缺口枚举"实现存在**系统性缺陷**：
1. **方向偏**——把"诊断充分性"做成了"临床工作流完整性"，大量追问分期/分级/不可获取信息；
2. **无门控**——只要 gaps 非空就追问，没有"这个 gap 值得问吗"的判断；
3. **输入不可靠**——基于可能丢信息的 summary 判断缺口，把对的诊断逼成错的。

其中 **P0 三项是低成本高收益的规则层修复**（不改模型，只改规则树 + 一层过滤），预计能同时压降死循环和"把对的改错"两类破坏。
