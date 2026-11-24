# P0 规则修复三件套 · 重跑验证报告

> 提交 `47bad0a` 落地了三件套代码修复，本报告给出 60 例重跑的确定性对比结论。
> 重跑输出目录：`refine_v2`（REJECT 修复）、`refine_cv_qwen_v2`（decisive 门控 + 限轮数 ≤2）。

## 一、判定口径

与 `finalize_60.py` 一致：`rule_judge` 返回 `True` 判对，`False/None` 判错
（`None` 经人工复核全部为真实错误，不翻判）。同一 `(prediction, ground_truth)` 对
规则层永远给出相同判定，无 LLM judge 噪声。

## 二、60 例最终准确率

| 方法 | 总数 | 正确 | 准确率 |
|---|---:|---:|---:|
| refine（旧基线） | 60 | 31 | 51.7% |
| refine_v2（REJECT 修复） | 60 | 30 | 50.0% |
| refine_cv_qwen（旧同模型） | 60 | 34 | 56.7% |
| refine_cv_qwen_v2（门控+限轮） | 60 | 34 | 56.7% |

**分段**：

| 方法 | 旧 30 例 (id 0–29) | 新 30 例 (id 30–59) |
|---|---:|---:|
| refine（旧基线） | 60.0% | 43.3% |
| refine_v2（REJECT 修复） | 56.7% | 43.3% |
| refine_cv_qwen（旧同模型） | 73.3% | 40.0% |
| refine_cv_qwen_v2（门控+限轮） | 70.0% | 43.3% |

## 三、逐例 diff

### 修复一：REJECT 信号修复（refine → refine_v2）

**修好 8 例 / 改坏 9 例**，McNemar `b=9, c=8, p=1.000`（不显著），净 -1 例。

典型修好（REJECT 原本被静默丢弃、错判为 PASS，现被追问拉回）：
- `medqa_13` Meconium ileus → **Hirschsprung disease** ✓
- `medqa_20` Occult type 2 DM → **CMV retinitis** ✓
- `medqa_52` Biliary cystadenocarcinoma → **Pancreatic ductal adenocarcinoma** ✓

典型改坏（REJECT 触发追问后，把本来对的诊断改错）：
- `medqa_7` Congenital Rubella Syndrome → GJB2-related hearing loss ✗
- `medqa_18` Uterine rupture → Severe placental abruption ✗
- `medqa_29` Disseminated gonococcal infection → Septic arthritis ✗

### 修复二：decisive 门控 + 限轮数 ≤2（refine_cv_qwen → refine_cv_qwen_v2）

**修好 7 例 / 改坏 7 例**，McNemar `b=7, c=7, p=0.789`（不显著），净 0 例。

典型修好：
- `medqa_18` Placental abruption → **Uterine rupture** ✓
- `medqa_38` Insufficient data → **Duodenal Atresia** ✓
- `medqa_48` von Willebrand → **Hemophilia A** ✓

典型改坏：
- `medqa_2` Hirschsprung → Intussusception ✗
- `medqa_5` Pes anserine bursitis → Tibial tuberosity stress injury ✗
- `medqa_13` Hirschsprung → Meconium ileus ✗

## 四、核心结论

1. **三件套是"正确性修复"，不是"性能修复"。** 它修复了上游 `nanshine/EID-Benchmark`
   静默丢弃 `REJECT` 信号的 bug，让行为符合设计意图；但在 60 例规模下**没有带来
   统计显著的准确率提升**（REJECT 修复 -1.7pp，门控+限轮 0pp，均 p>0.05）。

2. **REJECT 修复反而暴露了"追问有害"这一更深层问题。** REJECT 从"静默 PASS"改为
   "继续追问"后，追问机制有了把对的诊断改错的机会（改坏 9 例 ≥ 修好 8 例）。这与
   之前 gap 机制分析结论一致——**追问本身的边际价值为负**（"追问本身只剩改对 1 例"）。

3. **由此反推原文"静默丢弃 REJECT"可能是一个(无意的)性能保护**：不追问反而避免了
   把已答对的诊断改错。修复它之后，必须配套"追问置信度门控"，否则净效果为负。

4. **decisive 门控 + 限轮数方向正确但力度不足**：7 修好 7 改坏完全对冲，说明仅靠
   `decisive` 布尔门控 + `gather_count < 2` 还不够，未能把"有害追问"压到 0。

## 五、下一步建议

1. **保留 REJECT 修复**（正确性优先），但追问必须加高置信门控：仅当 verifier 高置信
   指出**具体** gap **且给出替代诊断**时才追问，否则直接收口。
2. **限轮数从 2 降到 1**，进一步压缩追问窗口。
3. **对"追问后诊断变化"加采纳门槛**：新诊断需比原诊断在 verifier 评分上显著更优
   （而非仅"不同"）才允许替换，避免"为变而变"。
4. 在更大规模（≥200 例）上重测，当前 60 例（含 30 例旧样本红利）信噪比不足。

## 六、产物清单

- 重跑结果：`benchmarks/eid-benchmark/results/medqa/refine_v2/`（60 例）
- 重跑结果：`benchmarks/eid-benchmark/results/medqa/refine_cv_qwen_v2/`（60 例）
- 收口脚本：`src/finalize_p0_compare.py`
