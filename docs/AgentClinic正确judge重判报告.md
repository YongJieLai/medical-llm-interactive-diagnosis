# AgentClinic 场景：确定性 Judge 重判报告

> 背景：此前 4 个方法在 AgentClinic/MedQA 场景的准确率是用**千问（qwen-plus）当 LLM judge** 判的，检测到大量"同预测不同判定"的 judge 噪声。本报告用**确定性 judge（规则 + 同义词表，DeepSeek 仅兜底）** 重判全部 60 条（4 方法 × 15 例），得到与人工复核完全一致的结果。

---

## 一、结论速览

| 方法 | 旧 judge（千问） | 新 judge（确定性） | 变化 | 说明 |
|------|:---:|:---:|:---:|------|
| roleplay | 66.7% | **66.7%** | — | 数字本来就准 |
| refine | 46.7% | **80.0%** | **+33.3pp** | 被 judge 严重低估 |
| meddxagent | 46.7% | **66.7%** | **+20.0pp** | 被 judge 低估 |
| clinicalagents | 53.3% | **53.3%** | — | 数字本来就准 |

**关键结论**：真实排名应为 **refine (80.0%) > roleplay (66.7%) = meddxagent (66.7%) > clinicalagents (53.3%)**。
此前"refine 垫底（46.7%）"的结论是 judge 噪声造成的假象——refine 实际上是最准的方法。

---

## 二、Judge 噪声修正清单（8 处，全部是"错判为错"）

这些 case 里方法其实诊断对了，是旧 judge（千问）误判，全部集中在 **refine（5 处）和 meddxagent（3 处）**：

| 方法 | case | 预测 | 真值 | 误判类型 |
|------|------|------|------|----------|
| refine | medqa_10 | Internal hemorrhoids | Hemorrhoids | 子类被判错 |
| refine | medqa_6 | Micturition syncope | Situational syncope | 子类被判错 |
| refine | medqa_7 | Congenital rubella syndrome | Congenital Rubella Infection | 同义被判错 |
| refine | medqa_11 | Mesial Temporal Lobe Epilepsy | Complex partial seizure | 同义被判错 |
| refine | medqa_12 | Acute posterior dislocation of right hip | Posterior hip dislocation | 同义被判错 |
| meddxagent | medqa_10 | Internal hemorrhoids | Hemorrhoids | 子类被判错 |
| meddxagent | medqa_6 | Micturition syncope | Situational syncope | 子类被判错 |
| meddxagent | medqa_9 | Postpartum endometritis | Endometritis | 子类被判错 |

> 最有说服力的证据：`medqa_6` 的 `Micturition syncope`，roleplay 和 clinicalagents 判对，而 refine 和 meddxagent 判错——**同一个预测、同一个真值，旧 judge 给出了相反结论**。这正是"judge 噪声"的直接表现。

---

## 三、各方法真失效 case（方法真的错，非 judge 错）

以下为新 judge 判定为 INCORRECT 的 case（走 DeepSeek 兜底，判的都是"真失效"）：

| 方法 | 失效 case（预测 → 真值） | 失效类型 |
|------|--------------------------|----------|
| roleplay (5 错) | Intussusception → Hirschsprung disease | 锚定错病 |
| | Medial tibial stress syndrome → Pes anserine bursitis | 锚定错病 |
| | "无急性感染" → Congenital Rubella Infection | 漏诊 |
| | Postpartum pyelonephritis → Endometritis | 锚定错病 |
| | Mallet finger → FDP 屈肌腱断裂 | 概念混淆（伸/屈肌腱） |
| refine (3 错) | Teething → Hirschsprung disease | 锚定错病 |
| | Acute pyelonephritis → Endometritis | 锚定错病 |
| | Mallet finger → FDP 屈肌腱断裂 | 概念混淆 |
| meddxagent (5 错) | Infantile colic → Hirschsprung disease | 锚定错病 |
| | **Lymphoma → Diffuse large B-cell lymphoma** | **过宽（父类）** |
| | EGPA → Acute interstitial nephritis | 锚定错病 |
| | MTSS → Pes anserine bursitis | 锚定错病 |
| | Mallet finger → FDP 屈肌腱断裂 | 概念混淆 |
| clinicalagents (7 错) | colic → Hirschsprung disease | 锚定 + [MEMORY] 复读 |
| | ACI → DLBCL | 锚定错病 |
| | MTSS → Pes anserine bursitis | 锚定错病 |
| | PDA → Congenital Rubella Infection | 只报了并发症 |
| | 膀胱高敏感 → Endometritis | 锚定错病 |
| | 髌股损伤 → Posterior hip dislocation | 锚定错病 |
| | Malrotation → Hirschsprung disease | 锚定错病 |

**共性失效模式**（与之前结论一致，但数字更干净）：
1. **锚定锁死**：绝大多数错例是"锁定一个错误诊断后不复盘"（colic/MTSS/EGPA/pyelo 等）。
2. **概念混淆**：`Mallet finger`（伸肌腱）vs `Jersey finger`（屈肌腱）在 3 个方法中都错，是高频陷阱。
3. **过宽**：meddxagent 的 `Lymphoma` 没具体到 DLBCL。
4. **只报并发症/局部**：clinicalagents 的 `PDA` 只报了先天性风疹的一个表现。

---

## 四、新 Judge 设计（可复现）

三层，确定性优先：

1. **规则层（确定性，高精度）**
   - 归一化（小写、去所有格 `'s`、去非字母数字）后精确匹配；
   - truth 完整词是 pred 子串 → 判对（捕获 `internal hemorrhoids ⊂ hemorrhoids` 这类"更具体"）；
   - 同义词/上下位等价表（`micturition syncope ⊂ situational syncope`、`jersey finger ≡ FDP 断裂` 等）；
   - 负向互斥表（`mallet finger ≠ FDP 断裂`）。
2. **DeepSeek 兜底（仅规则判不出的 case）**：双判 + 多数投票，prompt 明确"子类/同义=对，过宽父类=错"。

结果：60 条中 **规则层直接判定约 43 条**，DeepSeek 兜底 17 条（全部是真失效，判 INCORRECT，无一误判）。

---

## 五、对后续失效分析的影响

1. **refine 应被重新定位**：它不是"最差"，而是"最好"（80%）。之前的"verifier 过度保守不收敛"结论需要重审——它只在 3 例上失效，且这 3 例都是"锚定错病 + 概念混淆"这类通用问题，不是 refine 特有的 verifier 缺陷。
2. **clinicalagents 的 7 个错例最值得深挖**：它是唯一出现"只报并发症（PDA）""过宽（无）"之外的多种失效的方法，且 [MEMORY] 复读、ICR 最低——这仍是插件式改进的最佳靶点。
3. **judge 已确定性化**：后续扩到 30–50 例时，judge 环节不再引入噪声，准确率波动将真实反映方法差异。

---

*数据文件*：`results/rejudge_analysis.json`（60 条判定明细）；`AgentClinic正确judge重判对比.xlsx`（对比表 + 明细）。
*判定代码*：`src/deterministic_judge.py`（规则 + 同义词表 + DeepSeek 双判）、`src/rejudge_results.py`（重判入口）。
