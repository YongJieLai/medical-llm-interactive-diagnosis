# wrong_entity 深挖：90% 是真错实体，缺的是「判别性证据」而非「广覆盖」

> 状态：分析完成（2026-11-27）
> 前置：`diagnosis-debias-results.md`（cover 底座 70 错题，wrong_entity 41 例 59% 是最大头）、`cover-collection-fairness.md`。
> 本文回答：这 41 例 wrong_entity 里，多少是「真·同系统错实体」、多少是「实体粒度不同被判分冤枉」——决定下一步打采集还是修判分。

## 1. 结论（一句话）

**cover 底座的 wrong_entity 里，90%（35/39）是「真·同系统错实体」，只有 10%（4/39）是「实体粒度不同」被判分冤枉。判分口径基本公平（冤枉量仅 +2%），不用修判分，主战场在「同系统内的判别性证据采集」。**

## 2. 数字

对 cover 底座 70 例错题用 DeepSeek 做两段式分类（先定错误类型，再对 wrong_entity 判粒度）：

| 错误类型 | 例数 | 占比 |
|---|---|---|
| off_system | 23 | 32.9% |
| **wrong_entity** | **39** | **55.7%** |
| overdiagnosis | 4 | 5.7% |
| underdiagnosis | 2 | 2.9% |
| give_up | 2 | 2.9% |

> 注：另一次独立分类（`diagnosis-debias-results.md`）得 wrong_entity 41 / off_system 18。两次边界略有抖动，但 **wrong_entity 稳定在 39–41 例（55–59%）**，是绝对大头，结论稳健。

wrong_entity 子类拆分：

| 子类 | 例数 | 含义 |
|---|---|---|
| **true_wrong（真错实体）** | **35** | 同系统内选错了另一个「真正不同的病」 |
| granularity_mismatch（粒度不同） | 4 | 与 truth 同一临床谱系，只是粗/细颗粒不同，被判分误杀 |

判分口径评估：

| 口径 | 数值 |
|---|---|
| 当前准确率 | 130/200 = **65.0%** |
| 判分偏严量（granularity_mismatch 4 + judge_false_negative 0） | 4 例 |
| **修正后真实准确率** | **134/200 = 67.0%（+2.0%）** |

**判分偏严只影响 2%**——之前担心的"真实准确率被低估"不成立，判分是公平的。

## 3. 4 例「粒度不同」其实也站不住脚

连这 4 例都值得商榷，真实冤枉量可能 <2%：

| truth | pred | 为什么"粒度不同"也勉强 |
|---|---|---|
| Bowen's Disease | Bowenoid Papulosis | 两者病理/HPV 关联/恶性潜能都不同，临床常需区分 |
| Labyrinthitis | Acute bilateral vestibular hypofunction | 前者含炎症，后者是功能性，病因不同 |
| Physiological neonatal jaundice | Breastfeeding jaundice | 时间窗与病因不同 |
| Pemphigus vulgaris | Drug-induced pemphigus foliaceus | 一个是自身免疫、一个是药物诱导，亚型不同 |

## 4. 35 例「真错实体」的模式：教科书级鉴别诊断对

逐例看这 35 例，规律极其一致——**每一对都是"同系统内需要一条判别性证据才能分开的病"**，且这条证据往往单条、决定性：

| 案例对（truth → pred） | 缺失的判别性证据 |
|---|---|
| Hemophilia → von Willebrand disease | 因子 VIII vs vWF 抗原/活性；关节出血 vs 黏膜出血 |
| Glanzmann's Thrombasthenia → vWD | 血小板聚集功能（ristocetin 无聚集） |
| Takotsubo → acute decompensated HF（出现 2 次） | 冠脉造影无阻塞 + 心尖球样变 |
| Steven-Johnson syndrome → Herpes zoster | 黏膜受累 + 药物史 + 皮损形态 |
| Keratoacanthoma → hypertrophic actinic keratosis | 皮损数周快速生长 + 病理角化结构 |
| Exertional heat stroke → Neuroleptic Malignant Syndrome | 抗精神病药用药史 vs 运动暴露；肌强直 |
| Lambert-Eaton → NSCLC | 电生理重复刺激递增（LEMS 本体） |
| Ruptured ovarian cyst → ectopic pregnancy | β-hCG 是否阳性 |
| Pseudocyesis → biochemical pregnancy（2 次） | β-hCG 阴性 + 无宫内孕囊 |
| Psoriatic arthritis → Gout | 尿酸/晶体 vs 银屑病史 + 指炎 |
| Silent thyroiditis → thyroid nodule | 甲状腺功能谱 vs 影像结构 |
| Hirschsprung's disease → meconium ileus | 直肠活检无神经节 vs 囊性纤维化相关 |
| Thoracic aortic rupture → hemothorax | CT 血管造影主动脉破裂征 |

## 5. 关键洞察：方案 A 采到的是「共性证据」，不是「判别证据」

这是本次深挖最有价值的发现，它解释了方案 A 的一个隐含盲区：

> 方案 A 把覆盖率从 31% 抬到 50%，但采到的大量证据是**"这个系统里所有病都会有的共性症状"**（咳嗽、发热、胸痛、出血史），而不是**"能把同系统两个候选病分开的判别性证据"**（β-hCG、尿酸晶体、冠脉造影、vWF 抗原）。

所以覆盖率的提升，并不等价于"判别信息"的提升。剩下的 35 例 wrong_entity，本质是：

> 医生已经问到了"出血/心衰/皮疹/甲亢"这个系统，但**没有采到那条把系统内两个候选病切开的关键化验/影像/病史**。

## 6. 对下一步的启示（开放问题，尚未有答案）

这 35 例指向的真正战场是 **「同系统内的判别性证据定向采集」**。但它与之前所有失败方案只有一线之隔，必须先想清楚再动手：

| 路径 | 锚点 | 已知结局 |
|---|---|---|
| P0 / 方案 B（top-2 → 问判别问题） | 假设 | 负结果，假设错则锁死 |
| 方案 A（8 维广覆盖） | 证据类别 | 正结果，但采到的是共性证据 |
| **未知：判别性证据采集** | **证据的"判别力"** | **未做** |

关键张力：判别性证据天然依赖"当前候选集"，而"候选集"一旦显式化就滑回假设驱动。**能否在不锚定 top-2 的前提下，识别出"哪些证据维度此刻还是共性、需要补判别点"——这是真正的 research gap，也是这条线能不能和 P0/方案 B 划清界限的地方。**

候选思路（待验证，均未实现）：
1. **判别力标注**：在 8 维框架里，给每个维度标注"共性 vs 判别"两类，要求医生优先采判别性子维度（不提供具体问题，只给维度提示）。
2. **框架分层**：显式把"已采证据按判别力分层"，让医生看到"哪些维度只有共性证据、还缺判别证据"（证据驱动，非假设驱动）。
3. **先证伪再确立**：当某维度证据无法区分系统内多候选时，输出"证据缺口声明"而非硬猜（衔接 give_up 的正面用法）。

---

## 7. 附：分析方法

- 脚本 `src/classify_wrong_entity.py`：DeepSeek 两段式分类，temperature=0，json_mode。
- 输入：summary + truth + pred；输出：error_type + granularity + reason，逐例落盘 `results/wrong_entity_analysis_cover.json`。
