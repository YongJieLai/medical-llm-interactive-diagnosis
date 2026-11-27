# 交付文档 — 医疗 LLM 交互式诊断：主战场定位与度量标准修正

> 交付时间：2026-11-27 ｜ 状态：主战场已定位（采集层关键检查定向采集），度量标准正在从 confirmatory 重定义为 discriminating。
> 本文件是当前最完整的进展快照，取代 `HANDOFF.md`（2026-11-24）作为第一入口。

---

## 0. 一句话定位

本项目做**医疗 LLM 交互式诊断的失效分析**，目标不是刷 SOTA，而是回答一个问题：

> **交互式诊断的正确率天花板在哪里？失效发生在"诊断器"还是"证据采集器"？**

本阶段（11-25 至今）的结论链：

```
oracle 实验 → 诊断器非瓶颈（喂完整证据可达 87.5%，在线采集仅 64.5%）
     ↓
主战场 = 采集层的「关键证据定向采集」
     ↓
行为分析 → 不是"查多少"，是"查什么"（噪音检查 vs 判别检查）
     ↓
过程归因 → 61.9% 是环境不给（reporter 逐字检索），非模型缺陷
     ↓
面板补全 → 负结果：环境瓶颈消除后准确率纹丝不动，主战场回到"定向采集"
     ↓
度量标准修正 → "关键检查"旧定义（confirmatory）掺水，正重定义为 discriminating
```

---

## 1. 核心结论（带证据，区分"干净"与"作废"）

### 结论 1：诊断器非瓶颈（**干净证据**，不依赖任何"关键检查"定义）

同一 qwen-plus 诊断器、同一 prompt，唯一变量是喂给它的证据：

| 变体 | 证据来源 | 准确率 |
|---|---|---|
| **oracle_full** | 开卷喂全部事实 | **87.5%** |
| oracle_exam | 开卷只喂检查结果 | 70.5% |
| **在线交互采集（方案A）** | 医生在 16 轮内自己采 | **64.5%** |

**交互采集距 oracle_full 有 23pp 空间**，而诊断器本身能到 87.5% → **诊断器不是瓶颈，采集层才是**。

> ⚠️ `oracle_key`（喂"关键事实"）＝80.0% 这一档**已作废**——它依赖旧的"关键检查"定义（见 §3），待判别性定义重标注后重算。`oracle_full` / `oracle_exam` / `64.5%` 三档是干净的。

### 结论 2：不是"查多少"，是"查什么"（**干净证据**）

| 方法 | 检查占总交互 | 关键检查覆盖率（⚠️旧定义） |
|---|---|---|
| 基线 refine_cv | 66.9% | 42.3% |
| 方案A refine_cv_cover | 42.1% | 61.1% |

基线查得最多、关键检查覆盖率最低；方案A查得更少、关键检查覆盖率最高。**真正的差异是"噪音检查"（体温/血压/血常规）vs"判别性检查"（ptosis、肌力、抗体）。**

### 结论 3：61.9% 的"没采到"是环境不给，非模型缺陷（过程归因）

对没采到的判别性检查，回看模型每一轮的 `[THOUGHT]` 决策上下文，机制分布：

| 机制 | 占比 | 性质 |
|---|---|---|
| 环境不可用（请求了但 reporter 不给） | 61.9% | 非模型缺陷 |
| 轮次耗尽（想到但排后面） | 16.4% | 采集规划 |
| 方向没想到 | 9.4% | 真知识缺口 |
| 冗余闭合（已用别的证据确诊） | 9.0% | 当时最优 |
| 优先级错配 | 2.5% | 决策取舍 |

根因：reporter 是 closed-world **逐字检索**，医生请求的检查不逐字命中 `exam_facts` 面板就返回 "Not specified"。当时面板稀疏（均值 14.87 条/例）。

### 结论 4：面板补全 = 干净负结果（**干净证据**）

把 200 例检查面板全部补全（exam_facts 14.87→39.5 条/例）后重跑：

| 指标 | 稀疏面板 | 补全面板 | 变化 |
|---|---|---|---|
| **准确率** | 64.5% | 64.5% | **+0.0pp** |
| 检查命中率 | 27.4% | 48.5% | +21.1pp |
| 每例采到检查 | 7.6 | 14.3 | +88% |
| 关键检查采到数（⚠️旧定义） | 569 | 555 | −14 |

**消除环境瓶颈后准确率纹丝不动** → 证伪"环境瓶颈是主战场"。医生下检查的次数没变（~6次/例），补全面板只是让每次请求批量返回更多"常规噪音检查"。**主战场回到"关键检查的定向采集"。**

### 结论 5：度量标准必须重定义（本阶段最重要的方法论成果）

见 §3。旧的"关键检查"定义（confirmatory）有结构性缺陷，导致结论 2、3 里所有"关键检查覆盖率"数字作废，需用判别性定义重算。

---

## 2. 实验总账（正/负结果全清单）

| # | 方案 | 结果 | 状态 |
|---|---|---|---|
| 1 | 方案A：去效率优先 + 8维证据框架（refine_cv_cover） | **+6.5pp（64.5%）** | ✅ 唯一正结果 |
| 2 | P0：主动 top-2 假设优先 | −4.5pp | ❌ |
| 3 | 两阶段探索-利用 K=4/6/8 | 54–55.5% | ❌ |
| 4 | 方案B：采集规划器 | −10 例 | ❌ |
| 5 | 诊断层去偏置方案1（分级强制） | −14 例 | ❌ |
| 6 | 方案C1：判别性证据优先 | −20 例（p=0.011） | ❌ |
| 7 | 面板补全 61→200 例 | +0.0pp | ❌（负结果） |

**统一负结果机制**：采集层任何"优先级/选择/加速"措辞，都会退化成"少问、快结束"。LLM 采集器在运行时只能可靠执行"静态完整清单"，无法可靠执行"动态优先级排序"。这是 6 个负结果背后反复出现的规律。

---

## 3. 度量标准修正：confirmatory → discriminating（当前进行时）

### 3.1 旧定义的三个掺水点

旧 `key_facts.json` 把"关键检查"标注为"确诊该病的检查"，且开卷给正确答案：

1. **标准错位**：`confirmatory`（确诊）≠ `discriminating`（判别）。确诊检查是"已经猜到答案后的最后一锤"；判别检查才是"从候选中筛出正确项"。
2. **执行违背**：标注 LLM 连自己的标准都没执行——抽样发现把体温106°F、心率170、GCS 10、"aPTT 正常"等生命体征/正常值都标成了 KEY。
3. **循环依赖**：知道答案标"确诊检查"→ 采到确诊检查≈诊断对 → "覆盖率"与"正确率"几乎绑定，不是因果度量。

### 3.2 新定义（一句话）

> **判别性关键检查**：给定「主诉+病史」确定的鉴别诊断候选集 S（含正确诊断 D\*），某检查 e 是"关键"的，当且仅当 **e 的结果在 D\* 与 S 中至少一个竞争诊断之间产生不同预期**（能提供区分证据）。

### 3.3 防循环的设计（关键）

- **两个正交指标**：方向命中率（D\* 是否进 top-k 鉴别诊断 → 度量"知识缺口"）+ 判别性检查覆盖率（方向对时采到多少判别检查 → 度量"检查选择质量"）。
- **反例检验**：标注后必须验证"判别覆盖率"与"正确率"解耦——存在"没采判别检查却诊断对"和"采了却诊断错"的反例，才算度量真实；若 100% 同增同减，说明又循环了。

完整方案见 `docs/discriminating-key-exam-definition.md`。

### 3.4 作废与重算清单

| 数字 | 状态 |
|---|---|
| oracle_full 87.5% / oracle_exam 70.5% / 方案A 64.5% | ✅ 干净，保留 |
| oracle_key 80.0% | ⚠️ 作废，待判别性重标注后重算 |
| 关键检查覆盖率 42.3% / 61.1% / 59.6% | ⚠️ 作废，待重算 |
| 过程归因分布（env_unavailable 61.9% 等） | ⚠️ 机制分类本身独立，但分母依赖旧 key 定义，待复核 |

---

## 4. 产物清单

### 4.1 代码（`src/`，本阶段新增 6 个）

| 脚本 | 用途 |
|---|---|
| `run_oracle_ceiling.py` | oracle 三层上界实验（结论1） |
| `analyze_test_query_behavior.py` | 医生检查/询问行为分析（结论2） |
| `analyze_missed_key_exams.py` | 提取未采到的关键检查 |
| `classify_missed_key_exams.py` | 未采到检查的定性分类 |
| `process_attribute_missed_exams.py` | 过程归因（回看 `[THOUGHT]`，结论3） |
| `compare_panel_effect.py` | 面板补全前后对比（结论4） |
| `run_paper_modes.py`（改） | 支持 SHARD 分片 + 自定义数据集路径 |

另有 `summarize_missed_key_exams.py`、`summarize_process_attribution.py` 两个汇总脚本。

### 4.2 场景（`benchmarks/eid-benchmark/src/eid/scenarios/`）

| 场景 | 内容 |
|---|---|
| `refine_cv_cover.py` | 方案A（去效率优先 + 8维证据框架，正结果） |
| `refine_cv_freebudget.py` | 公平性消融 |
| `refine_cv_cover_plan.py` | 方案B（负） |
| `refine_cv_discrim.py` | 方案C1（负） |
| `refine_cv_debias.py` | 去偏置方案1（负） |

### 4.3 数据

- `benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl`：原始 200 例（14.87 exam_facts/例）
- `benchmarks/eid-benchmark/data/agentclinic_medqa_panel.jsonl`：**补全面板版**（39.5 exam_facts/例，200 例）
- `data/exam_panels/`：200 例检查面板（含审核 verdict）

### 4.4 结果

- `benchmarks/eid-benchmark/results/medqa/refine_cv_cover*/`：方案A 及变体的 200 例 trace
- `benchmarks/eid-benchmark/results/medqa/refine_cv_cover_panel/`：面板补全后的重跑 trace
- `results/coverage_analysis/`：key_facts、process_attribution、missed_key_exams 等分析产物

### 4.5 文档（`docs/`，本阶段新增 4 篇）

| 文档 | 内容 |
|---|---|
| `discriminating-key-exam-definition.md` | **度量标准重定义**（最重要） |
| `panel-completion-negative-result.md` | 面板补全负结果 |
| `process-attribution-reporter-bottleneck.md` | 过程归因 |
| `missed-key-exam-qualitative-analysis.md` | 未采到检查定性分析（已被后续推翻部分结论，以本文档为准） |

---

## 5. 下一步（按优先级）

1. **重做判别性标注**（P0）：按 `docs/discriminating-key-exam-definition.md` 重标 200 例，得到"方向命中率"+"判别性检查覆盖率"两个新指标，重算 oracle_key 和方案A 的覆盖率。
   - 待定参数：k（top-k 鉴别诊断，建议5）、严格版/降级版、是否 strong/weak 分层。
2. **反例检验**：验证新指标与正确率解耦，确认度量不再掺水。
3. **基于新指标重新归因**：方向命中率低 → 打知识缺口；判别覆盖率低 → 打检查选择引导。
4. **不泄露前提下的定向采集增强**：方案A 已验证 8 维框架有效（+6.5pp），下一步在判别性定义指导下加强"症状→判别检查"引导，但不得注入病例专属检查清单（泄露风险）。

---

## 6. 复现要点

### 环境与密钥
- Python 3.11；`config/api_keys.env` 配 `DASHSCOPE_API_KEY`（qwen）、`DEEPSEEK_API_KEY`（deepseek），**不入库**。
- 已知坑：tiktoken 沙箱 SSL 问题已用 `DummyTokenCounter` 绕过。

### 关键命令

```bash
cd /root/.codebuddy/artifact/2f8a7d30-9c21-4f0b-1d7e-faa84c687000/medical-llm-interactive-diagnosis

# oracle 上界
python3 src/run_oracle_ceiling.py

# 方案A 跑批（6 分片并行）
for i in 0 1 2 3 4 5; do \
  SHARD_ID=$i SHARD_TOTAL=6 python3 src/run_paper_modes.py refine_cv_cover 200 16 6 & \
done; wait

# 面板补全后的方案A
for i in 0 1 2 3 4 5; do \
  SHARD_ID=$i SHARD_TOTAL=6 DATASET_PATH=data/agentclinic_medqa_panel.jsonl OUT_SUFFIX=_panel \
  python3 src/run_paper_modes.py refine_cv_cover 200 16 6 & \
done; wait
```

---

## 7. 上游追溯

- EID-Benchmark：`https://github.com/NanshineLoong/EID-Benchmark.git`
- AgentClinic：`https://github.com/samuelschmidgall/agentclinic.git`
- MEDDxAgent：`https://github.com/nec-research/meddxagent.git`
- ClinicalAgents：`https://github.com/ZhuohanGe/ClinicalAgents-Code.git`
