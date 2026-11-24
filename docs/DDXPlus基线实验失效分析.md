# DDXPlus 基线实验失效分析（千问 qwen-plus）

## 一、实验设置

- **环境**：DDXPlus（合成病例，规则脚本患者精确查表回答，**无噪声**）
- **病例**：20 例，覆盖 20 个疾病，含易混淆的胸痛组（Stable/Unstable angina、NSTEMI/STEMI、Pericarditis、GERD）
- **模型**：千问 qwen-plus
- **方法**：vanilla（自由问诊基座）、MedKGI（LLM 生成 DDX → 信息增益选问）、SGUR（信念分布 + 熵减选问）、KnowGuard（证据池 + 弃权追问）
- **组合**：4 方法 × 20 病例 = 80 组合，全跑通

## 二、总成绩（正确率）

| 方法 | 正确率 | 平均轮数 | 平均问证据数 |
|---|---|---|---|
| **vanilla**（自由问诊） | **9/20 (45%)** | 19.3 | 19.2 |
| SGUR | 8/20 (40%) | 18.6 | 18.3 |
| MedKGI | 7/20 (35%) | 13.3 | 12.6 |
| KnowGuard | 5/20 (25%) | 20.0 | 20.0 |

**反直觉结论**：三个"结构化选问"方法在无噪声环境下**全部不如 vanilla 自由问诊**。结构化选问带来的"效率提升"（MedKGI 只问 12.6 个 vs vanilla 19.2 个）没有转化为准确率提升，反而因为**问得少而漏掉关键证据**。

## 三、病例级对错矩阵

| 病例（真值） | van | mkg | sgu | kng |
|---|---|---|---|---|
| Acute rhinosinusitis | ✓ | ✓ | ✓ | ✓ |
| Influenza | ✓ | ✓ | ✓ | ✓ |
| Bronchospasm / asthma | ✓ | ✓ | ✓ | ✓ |
| GERD | ✓ | ✓ | ✓ | ✓ |
| Stable angina | ✓ | ✗ | ✗ | ✗ |
| Pericarditis | ✓ | ✗ | ✗ | ✗ |
| Acute laryngitis | ✓ | ✗ | ✗ | ✗ |
| Unstable angina | ✗ | ✓ | ✓ | ✓ |
| Epiglottitis | ✗ | ✓ | ✗ | ✗ |
| NSTEMI / STEMI | ✗ | ✗ | ✓ | ✗ |
| Pneumonia | ✓ | ✗ | ✓ | ✗ |
| Pulmonary embolism | ✓ | ✓ | ✓ | ✗ |
| **Sarcoidosis / URTI / Chronic rhinosinusitis / COPD exacerbation / SLE / Panic attack / Cluster headache / Bronchiolitis** | ✗ | ✗ | ✗ | ✗ |

- **全对 4 例**：都是"常见、症状明确"的病
- **全错 8 例**：罕见病 / 症候群 / 非典型表现
- **分歧 8 例**：vanilla 独对 3 例（Stable angina、Pericarditis、Acute laryngitis），结构化方法独对 3 例（Unstable angina、Epiglottitis、NSTEMI）

## 四、核心失效模式（量化）

### 失效模式 1：锚定锁死（最致命，MedKGI 特有）

MedKGI 把"鉴别诊断生成"外包给 LLM 自由生成 DDX 候选集，信息增益只在候选集的证据池里选问。**一旦 LLM 首轮生成的候选集漏掉真病，后续 13 轮候选集漂移也跳不出来**：

| 指标 | 结果 |
|---|---|
| 真值疾病进过候选集 | 14 例 → 诊断对 7 例（50%） |
| **真值从未进候选集（锚定锁死）** | **6 例 → 诊断对 0 例** |

锚定锁死的 6 例：Sarcoidosis、Stable angina、Chronic rhinosinusitis、COPD exacerbation、Acute laryngitis、Cluster headache。

**典型机制**（Stable angina 病例）：首轮 DDX 候选集 = {肺水肿、肺炎、COPD、肺栓塞、NSTEMI}——全是呼吸系统病，心绞痛没进候选。此后 13 轮信息增益问的全是呼吸系统症状（E_161 呼吸困难、E_220、E_196…），而 Stable angina 的关键证据（E_53 胸痛、E_54/55 胸痛性质、E_66 运动诱发）**从头到尾没被问**，最终错诊为 Pulmonary embolism。

> **根因**：信息增益选问的公式没错，错在**先验（候选集）错了**。"垃圾进垃圾出"——在错误候选池里做再优的熵减选问，也只是更精准地偏离真病。

### 失效模式 2：胸痛组互相错诊（鉴别诊断失败）

Stable/Unstable angina、NSTEMI/STEMI、Pericarditis、PE、pneumothorax、pulmonary edema 这组"胸痛+呼吸困难"疾病，各方法**互相错诊**：

| 病例 | vanilla | medkgi | sgur | knowguard |
|---|---|---|---|---|
| Stable angina | ✓ 对 | → PE ✗ | → NSTEMI ✗ | → 肺水肿 ✗ |
| Unstable angina | → Pericarditis ✗ | ✓ 对 | ✓ 对 | ✓ 对 |
| NSTEMI/STEMI | → Panic attack ✗ | → 肺肿瘤 ✗ | ✓ 对 | → 气胸 ✗ |
| Pericarditis | ✓ 对 | → GERD ✗ | → PE ✗ | → GERD ✗ |

这些病共享核心症状（胸痛、呼吸困难），区分它们必须问**鉴别性证据**（胸痛是否运动诱发？是否放射到手臂？是否躺下加重？）。自由问诊的 vanilla 反而更容易问对，因为不预设候选框、自然追问细节。

### 失效模式 3：软终止过早 → 漏问（MedKGI）

MedKGI 有"候选集收敛就停"的软终止，导致它只问 12.6 个证据就下诊断（vanilla 问 19.2 个）。在无噪声环境下，**"问得多"本身就是优势**——因为真病的关键证据就藏在还没问到的那些里。MedKGI 的效率优势在简单环境里变成了漏问劣势。

## 五、结论

1. **在这个"干净"环境里，vanilla 自由问诊是强 baseline**（45%），结构化选问方法没能证明"选问"的价值，反而因候选集先验错误 + 软终止过早而更差。
2. **信息增益选问的生死线是"候选集（先验）的生成质量"**，不是选问公式本身。这是 MedKGI/SGUR 类方法的阿喀琉斯之踵：它们把最关键的"该考虑哪些病"外包给了 LLM 的开放生成，而 LLM 在罕见/非典型病例上会锚定在常见病。
3. **对后续 AgentClinic 实验的启示**：AgentClinic 有 LLM 患者噪声 + 主动开检查，难度更高，这些失效模式（锚定、鉴别诊断失败）预计会被放大，且会新增"检查决策失败""被患者噪声带偏"两类。这份 DDXPlus 基线可以作为"简单环境下方法已暴露的问题清单"，用于对照难环境下是否复现、是否加剧。

## 六、数据产物

- 原始过程日志：`/workspace/medical_llm_exp/results/runs_qwen.jsonl`（80 条，每轮含动作、候选集、所问证据、患者回答、诊断、耗时）
- 病例子集：`/workspace/medical_llm_exp/results/cases.json`
