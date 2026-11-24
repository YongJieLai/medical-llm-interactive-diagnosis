# AgentClinic 三方法（REFINE / MEDDxAgent / ClinicalAgents）失效分析报告

> 实验环境：EID-Benchmark 的 AgentClinic-MedQA 数据集，千问 qwen-plus，max-turns=10，15 病例。
> 对照：roleplay（自由问诊，作为下界）。
> 方法移植说明：REFINE 跑原版代码；MEDDxAgent / ClinicalAgents 为「机制级移植」（核心编排机制忠实保留，环境统一到 AgentClinic，见文末「移植保真度」）。

---

## 一、原始结果（直接看 judge 分数，会得出错误结论）

| 方法 | 正确率 | 信息覆盖率 ICR | 每病例 LLM 调用 |
|---|---|---|---|
| roleplay（下界） | **66.7%** | 0.280 | 15.8 |
| clinicalagents | 53.3% | 0.200 | 15.8 |
| refine | 46.7% | 0.308 | 21.3 |
| meddxagent | 46.7% | 0.312 | 23.8 |

表面结论：三个"先进方法"都不如 roleplay 下界。**这个结论是错的**，因为 judge 分数被系统性噪声污染。

---

## 二、核心发现 1：judge（LLM 裁判）本身是最大噪声源

客观检测到 **10 对「相同/相似预测却判定不一致」**：

| 病例（真值） | 方法A（判定） | 方法B（判定） | 预测内容 |
|---|---|---|---|
| medqa_10（Hemorrhoids） | roleplay（✅） | refine（❌）/ meddxagent（❌） | 都是 "Internal hemorrhoids" |
| medqa_6（Situational syncope） | clinicalagents（✅） | refine（❌）/ meddxagent（❌） | 都是 "Micturition syncope" |
| medqa_7（Congenital Rubella） | meddxagent（✅） | refine（❌） | 都是 "Congenital rubella syndrome" |
| medqa_3（DLBCL） | roleplay/refine（✅） | meddxagent（❌） | "Diffuse large B-cell lymphoma" vs "Lymphoma"（上位概念） |
| medqa_11（Complex partial seizure） | roleplay（✅） | refine（❌） | "Temporal lobe epilepsy" vs "Mesial TLE"（同一诊断） |
| medqa_12（Posterior hip dislocation） | roleplay（✅） | refine（❌） | "Posterior hip dislocation" vs "Acute posterior dislocation"（同一诊断） |

**同一句 "Internal hemorrhoids"，judge 在 roleplay 判对、在 refine/meddxagent 判错。** 这不是方法差异，是裁判不稳定。

**judge 噪声的两类来源**：
1. **上下位概念误判**：`Lymphoma`（淋巴瘤）被判错，但它是 `Diffuse large B-cell lymphoma` 的上位概念；`Micturition syncope`（排尿性晕厥）是 `Situational syncope`（情境性晕厥）的合法子类。
2. **同义表述判定不稳定**：同一诊断换种写法（加不加"acute"、加不加修饰语），判定结果漂移。

> 含义：**在 LLM-as-judge 场景下，"正确率"这个单一数字不可信**，必须逐例人工复核或做多数投票。这也呼应了之前 HealthBench Professional 被 Epoch AI 打"Flawed"的原因之一——模型裁判本身不可靠。

---

## 三、核心发现 2：人工复核 judge 噪声后，结论反转

把「预测语义正确但被判错」的病例人工纠正后：

| 方法 | 原始正确率 | 复核后正确率 | 变化 |
|---|---|---|---|
| roleplay（下界） | 66.7% | 66.7% | — |
| **refine** | 46.7% | **80.0%** | +33.3pp |
| meddxagent | 46.7% | 66.7% | +20pp |
| clinicalagents | 53.3% | 53.3% | — |

**修正后的真实格局**：
- **refine 的验证机制是有效的**（80% vs roleplay 66.7%），之前"不如 roleplay"是 judge 噪声造成的假象。
- **meddxagent 与 roleplay 持平**（66.7%）。
- **clinicalagents 确实最差**（53.3%），且这不是 judge 噪声，是真实失效。

---

## 四、各方法的真实失效模式（剔除 judge 噪声后）

### 4.1 ClinicalAgents：锚定锁死 + 工作记忆重复污染（最差，且是真实失效）

- **锚定锁死**（与 DDXPlus 实验发现的同源）：medqa_3（DLBCL）首轮假设锚定在"贫血/甲减/抑郁"，淋巴瘤从头到尾没进过假设空间，最后诊断成"慢性病性贫血"。medqa_13（Hirschsprung）锚定在"肠旋转不良"。
- **工作记忆重复而非增量更新**：每轮输出的 `[MEMORY]` 几乎原样复读上一轮，甚至把"患者不确定血色"这类"未解决项"反复写入而不推进。机制上"可变共享状态"的设想，在 qwen-plus 上退化成"复读机"。
- **ICR 最低（0.200）**：冗长的 `[MEMORY]` 输出挤占了本该用于问诊的注意力。

### 4.2 MEDDxAgent：patient profile 卡住 + 信息损耗

- **profile 增量更新停滞**：medqa_10 的 profile 从第 5 次更新起几乎不再变化（"Blood characteristics: Not specified" 反复出现），新信息没有被有效合并。
- **额外 LLM 调用成本高**：每病例 23.8 次调用（比 roleplay 多 50%），但正确率（复核后）只持平——**多花的算力没有换来收益**。
- 诊断质量本身尚可（复核后 66.7%），主要问题是**效率**和**profile 状态维护不可靠**。

### 4.3 REFINE：verifier 过度保守（over-conservatism）——复核后有效但有隐患

- **核心失效样本 medqa_10**：诊断其实已正确（内痔，anoscopy 确诊），但 verifier 反复判 "INCOMPLETE"，理由是"45 岁新发便血必须做结肠镜排除结直肠癌"，把医生反复打回补问结肠镜，诊断被无限推迟。
- 这是 **"追问"与"诊断闭合"的张力**：verifier 用"还能做更多检查"当成了"诊断不充分"，缺少"何时该停"的判断。在简单病例上，它会过度医学化、拖到轮数耗尽。
- 但复核后 refine 整体最好（80%），说明 verifier 在**真正缺证据的病例上确实能拦下错误**，只是**没有区分"证据已够"和"证据还缺"**。

### 4.4 全方法都错的困难病例（真困难，不是方法问题）

| 病例 | 真值 | 各方法错成 | 根因 |
|---|---|---|---|
| medqa_2 | Hirschsprung（先天性巨结肠） | 肠套叠/出牙/功能性腹痛 | 没问出"胎粪排出延迟+便秘"这条决定性线索 |
| medqa_9 | Endometritis（产后子宫内膜炎） | 肾盂肾炎/膀胱过敏 | 产后发热被误导向泌尿系 |

这两个病例**四个方法全错**，是真正需要"特定问诊路径"的困难病例，是后续方法改进的靶点。

---

## 五、结论

1. **别信单一 judge 分数**：LLM 裁判在上下位概念和同义表述上有显著不稳定，本次 15 病例里就有 10 对判定矛盾。这是比"方法好不好"更底层的问题。
2. **复核后，REFINE 有效（80%）、MEDDxAgent 持平（66.7%）、ClinicalAgents 最差（53.3%）**——之前的"都不如 roleplay"是噪声假象。
3. **真实的失效模式**：ClinicalAgents = 锚定锁死 + 工作记忆复读；MEDDxAgent = profile 卡住 + 算力浪费；REFINE = verifier 过度保守、不会"收手"。这三个正是可以针对性设计"插件"补的点。
4. **样本量太小（15 例）**：judge 噪声的影响被放大。下一步应扩到 30–50 病例，并对 judge 做**双判/多数投票 + 人工抽查**，才能得到可信结论。

---

## 附：移植保真度说明

| 方法 | 保留的核心机制 | 简化/丢弃的部分 |
|---|---|---|
| REFINE | 原版完整跑（summarizer→diagnostician→verifier 验证环） | 无 |
| MEDDxAgent | 模块化 agent 池（每轮选 HISTORY_TAKING/EXAMINATION/DIAGNOSIS）+ 患者 profile 增量更新 + 末轮强制诊断 | 丢弃 RAG 检索模块（AgentClinic 无外部知识库） |
| ClinicalAgents | 显式工作记忆（证据/假设/缺失证据/动作）+ 假设驱动证据获取 + 回溯 | MCTS 树搜索简化为单步 LLM 决策（动作空间仅 3 个，完整 rollout 成本不可接受）；丢弃经验记忆（无指南/历史病例库） |
