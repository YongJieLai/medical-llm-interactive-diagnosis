# Gap-Check 机制对比：原文门控 vs 我们的缺口枚举

> 纯离线分析，不重跑。对比 `refine`（REFINE 原文 Data Sufficiency 二元门控）与
> `refine_cv_qwen`（我们的三组件缺口枚举），两者都是 qwen-plus 同模型核对，唯一变量是 gap 机制。

## 一、结论摘要

| 发现 | 严重度 | 一句话 |
|---|---|---|
| **① 原文门控复现有个严重 bug：REJECT 信号被丢弃** | 🔴 高 | verifier 15 次明确"拒绝诊断"，代码只认 `INCOMPLETE`，`REJECT` 被静默当 PASS 放行，其中 13 次最终答错 |
| ② 两种 gap 的追问都接近无效 | 🟡 中 | 87 次追问累计：改对 1 例、改坏 3 例，其余全是无效追问（诊断不变） |
| ③ 缺口枚举是"恒 3 个凑数" | 🟡 中 | 103 次 verifier 调用里 103 次 gaps=3，仅 9 次 gaps=0 |
| ④ 机制 B 更高的准确率不来自 gap | 🟢 定性 | 来自 claim check（29 次打回）+ refutation（31 次触发），gap 是负资产 |

---

## 二、发现①：原文门控的 REJECT 信号被静默丢弃（最该修）

`refine` 场景的 verifier 实际输出了 **三种** 决策值，而 prompt 与代码不一致：

| verifier 输出 | 次数 | 代码如何处理 |
|---|---:|---|
| `PASS` | 29 | ✅ 接受 |
| `INCOMPLETE` | 24 | ✅ 触发追问 |
| `REJECT` | **15** | ❌ **被当成 PASS 静默接受** |

根因是 **prompt 与代码不一致**：

- `manager.py:291` prompt 写的是 *"force a decision (PASS or REJECT)"*
- `manager.py:301` 又写 *"Output 'INCOMPLETE' if critical clinical information is missing"*
- `refine.py:305` 代码只判断 `decision == "INCOMPLETE"` 才追问

**`REJECT` 的语义比 `INCOMPLETE` 更强**——它是"诊断错误/不可接受"，而 `INCOMPLETE` 只是"数据不足"。抽样 15 个 REJECT 的 FEEDBACK：

| id | REJECT 反馈（节选） | 最终结果 |
|---|---|---|
| 2 | "Diagnosis of UTI is not supported without confirmed urinalysis" | 错 |
| 14 | "diagnosis of mallet finger cannot be confirmed without…" | 错 |
| 20 | "occult type 2 diabetes mellitus is not sufficiently supported" | 错 |
| 37 | "Diagnosis cannot be confirmed without Thoracic spine MRI" | 错 |
| 42 | "Abdominal tuberculosis cannot be diagnosed on current evidence" | 错 |
| 50 | "Karyotype is mandatory to confirm or exclude Turner syndrome" | 错 |

**15 次 REJECT 里 13 次最终答错**——verifier 明明指出了诊断有问题，代码却全部放行了。
这直接拉低了 `refine`（原文门控）的准确率，也意味着**"原文门控"并没有被正确复现**。

---

## 三、发现②：两种 gap 的追问都接近无效

### 机制 A（原文门控）真实追问的 8 例

| 轨迹 | 结果 |
|---|---|
| id=19: 对 → 弃 | 🔴 改坏 |
| id=24/26/31/58: 对 → 对 | 无效 |
| id=28/36/46: 弃 → 弃 | 无效 |

**追问 8 例：改对 0，改坏 1，其余 7 例无效。**

### 机制 B（缺口枚举）诊断真正变化的 12 例

| 轨迹 | 结果 |
|---|---|
| id=29: 弃 → 对（DGI） | 🟢 改对（唯一 1 例）|
| id=38: 对 → 弃（Duodenal atresia）| 🔴 改坏 |
| id=48: 对 → 弃（Hemophilia→vWD）| 🔴 改坏 |
| 其余 9 例 | 措辞微调（对→对 / 弃→弃），无效 |

**87 次追问（A:8 + B:79）累计：改对 1、改坏 3，其余 83 次无效。**
追问的边际收益极低——gap check 作为"追问发动机"，两个版本都没能把"追问"转成"改对"。

---

## 四、发现③：缺口枚举是"恒 3 个凑数"

```
机制B verifier gaps 数量分布:
  gaps=0: 9 次
  gaps=3: 103 次   ← 几乎每次都是 3 个
```

prompt 要求 *"list the top 2-3 missing pieces of information"*，模型被诱导"必须列满 3 个"，
即使诊断已经明确（血清学确诊、影像确诊）也硬凑 3 个 gap。这是"无病呻吟"的来源，
也是死循环（13 例）的根源：`gaps[0]` 往往是"病理确诊/基因检测"这类 reporter 没有的死胡同。

---

## 五、发现④：机制 B 的增益来自 claim check + refutation，不是 gap

机制 B（56.7%）比机制 A（51.7%）高 5pp，但拆开看：

| 信号 | 触发次数 | 贡献 |
|---|---:|---|
| 坏 claim 打回 | 29 次（1 坏:20 / 2 坏:7 / 3 坏:2）| ✅ 主要增益 |
| refutation 触发 | 31 次 | ✅ 次要增益 |
| gap 追问 | 79 次 | ❌ 只改对 1、改坏 2 |

**gap 是负资产**：它贡献了最多的追问次数（79 次）却几乎不带来正确率提升，
还引入了 2 例"把对的改错"和 13 例死循环。

---

## 六、改进方案（按优先级）

### P0-1（最关键）：修复原文门控的 REJECT 信号丢失

把 `refine.py` 的追问条件从 `decision == "INCOMPLETE"` 改为 `decision in ("INCOMPLETE", "REJECT")`，
并统一 prompt 术语（去掉 "PASS or REJECT" 的混乱表述）。

**预期收益**：机制 A 里被丢弃的 15 次 REJECT（13 次答错）有机会被重新诊断纠正，
这是当前"捡回门控语义"最直接、零成本（只改一行 + 重跑）的收益。

### P0-2：给缺口枚举加"门控"，治"恒 3 个凑数"

把机制 B 的 `gaps 非空就追问` 改成**两段式**：

1. verifier 先输出二元判定 `decisive / not_decisive`（复用原文"够不够"语义）；
2. 只有 `not_decisive` 时才列 gap，且限制 gap 必须"能区分当前诊断 vs 明确替代诊断"。

这样能砍掉大部分无效追问（gaps=3 的 103 次里，很多是诊断已明确还在凑数）。

### P0-3：限制追问轮数 ≤ 2

死循环 13 例的直接解法，规则层加个轮数上限即可。

### P1：验证"修好门控 + claim check"二组件（不带无脑 gap）

推测最优配置是 **原文门控（修好 REJECT）+ claim check**：
保留 claim check 的 29 次打回增益，用正确的二元门控替代无脑 gap 枚举。
有理由冲过当前 56.7%。

---

## 七、一句话结论

> 两个 gap check 版本都有病：**原文版丢了 REJECT 信号（实现 bug），我们的枚举版无脑凑 3 个 gap（设计缺陷）**。
> 但两者共同的教训是——**"追问"本身几乎没有用（87 次追问只改对 1 例）**。
> 真正的价值在 claim check（证据真伪核对）和 refutation（反证），gap 应该从"追问发动机"
> 降级为"仅在诊断不充分时的一个温和提示"，而不是无条件触发多轮追问。

---

*数据文件：`results/gap_mechanism_compare.json`（60 例逐例诊断轨迹与决策值）*
