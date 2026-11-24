# HANDOFF — 医疗 LLM 交互式诊断（CEV-GR）项目交接

> 交接时间：2026-11-24 ｜ 状态快照：扩样本 60 例跑批已全部完成，机制归因与 F3 多模型表征待做。

---

## 0. 一句话定位

本项目做**医疗 LLM 交互式问诊/诊断的失效分析**，并把文献方法移植到 AgentClinic（MedQA）场景做对照。
目标不是刷 SOTA，而是：**检查现有方法的失效场景 + 提出/验证零训练插件 CEV-GR + 沉淀我们自己的可复现发现 F3（跨模型语义不对齐）**。

---

## 1. 研究脉络（时间线，帮助快速建立上下文）

| 阶段 | 做了什么 | 产出文档 |
|------|---------|---------|
| 调研 | 医疗 benchmark 清单、主动问诊/证据验证方法精读 | `docs/医疗大模型评测基准清单.xlsx`、`docs/医疗LLM*精读.md` |
| 基线 | DDXPlus 规则患者跑通，发现太简单 | `docs/DDXPlus基线实验失效分析.md` |
| 移植 | 把 REFINE / MEDDxAgent / ClinicalAgents 移植到 AgentClinic（EID-Benchmark） | `docs/AgentClinic三方法失效分析报告.md`、`docs/AgentClinic30例失效模式分析.md` |
| 失效分析 | 四方法失效拆解：锚定 37% / 正确假设被丢弃 24% / 未采集 14% / 读反 6% | 同上 |
| 插件设计 | CEV-GR（跨模型证据核对 + 缺口驱动反证） | `docs/refine_cv插件设计.md`、`docs/refine_cv对比与移植矩阵.md` |
| 核心发现 | **跨模型解耦假设被证伪 → F3 跨模型语义不对齐** | `docs/CEV-GR插件对比实验结果.md`、`docs/CEV-GR消融与稳健性实验结果.md` |
| 扩样本 | 30 → 60 例（5 方法），**跑批已全部完成** | 本仓库 results（见 §4） |

---

## 2. 核心发现（已确立，别推翻重来）

### 发现 1：F3 跨模型语义不对齐（本项目最接近独立贡献的点）

**跨模型核对（DeepSeek 核对 qwen）= 40% < 基线 refine = 60% < 同模型核对（qwen 核对 qwen）= 73.3%。**

根因不是"DeepSeek 弱"，而是**模型间语义边界不一致**：核对模型把生成模型的「临床推理概括语言」（如 *"classic ulcer morphology — pathognomonic for PG"*）误判成「编造证据」，反复打回直到诊断崩溃。

证据链（四组，30 例，确定性 judge）：

| 组合 | 生成端 | 核对端 | 准确率 |
|------|:---:|:---:|:---:|
| 跨模型 | qwen | DeepSeek | 40.0% |
| 强跨模型 | qwen | qwen-max | 66.7% |
| 同模型 | qwen | qwen | **73.3%** |
| 同模型·DS | DeepSeek | DeepSeek | **76.7%** |

`qwen-max`/`全 DeepSeek` 两组交叉验证了「DeepSeek 弱」的替代解释不成立。

### 发现 2：同模型证据核对有效；单一组件无效、三机制协同才有效

30 例消融：`check` 63.3% / `gap` 56.7% / `refute` 56.7%，**单组件都不显著超过基线（60%）**；完整版（三机制齐）73.3%（+13.3pp）。
有 4 个 case 完整版对、三个单组件全错（协同证据），另有 2 个 case 基线对、单组件破坏、完整版保住（单组件有副作用）。

### 发现 3：judge 噪声的历史修正（重要，避免混淆数字）

- 早期报告 refine = 73.3% 是 **DeepSeek 兜底 judge 虚高**；规则层 + 人工复核后真实 refine 基线 = **60.0%**。
- 当前统一口径：`deterministic_judge.py` 的 `rule_judge`（规则层：精确/子串/同义词表）+ 规则判不出的人工复核（或 DeepSeek 双判多数投票兜底）。
- **ExactMatch 的 `result_new.json` 是占位，不可作为最终数字。**

---

## 3. 当前任务状态（对应之前的三个任务）

| 任务 | 状态 | 说明 |
|------|------|------|
| **① 扩样本到 60 例** | ✅ 跑批完成 | 5 方法（refine / refine_cv_qwen / check / gap / refute）各 60 个 trace（id 0–59），`record/` 已就绪 |
| **② F3 多模型表征** | ✅ 代码就绪，待跑 | `refine_cv_glm` 场景 + 生成端 GLM 已写好注册；`ZHIPU_API_KEY` 已配到本地 `config/api_keys.env`（**不入库**） |
| **③ 机制归因** | ⏳ 待做 | 依赖 ① 的最终判定 + ② 的多模型组合 |

---

## 4. 扩样本 60 例的当前数字（规则层，本次交接前新算）

用 `python3 src/compare_variants.py` 得到（**规则层** `rule_judge` 确定 CORRECT 占比；「不确定」暂按错）：

| 方法 | 60 例规则层正确/总 | 规则层准确率 | 不确定数 |
|------|:---:|:---:|:---:|
| refine（基线） | 28/60 | 46.7% | 31 |
| **refine_cv_qwen（同模型完整）** | **32/60** | **53.3%** | 28 |
| refine_cv_qwen_check | 29/60 | 48.3% | 31 |
| refine_cv_qwen_gap | 28/60 | 46.7% | 32 |
| refine_cv_qwen_refute | 28/60 | 46.7% | 32 |
| refine_cv（跨模型 DS，仅 30 例） | 12/30 | 40.0% | 17 |
| refine_cv_qwenmax（仅 30 例） | 20/30 | 66.7% | 10 |
| refine_cv（全 DeepSeek，仅 30 例） | 23/30 | 76.7% | 7 |

> ⚠️ 关键提醒：
> 1. 以上是**规则层**数字，**「不确定」case 尚未人工复核**，最终数字会更高（30 例时人工复核把部分不确定判对）。
> 2. 新增 id 30–59 这批病例的规则层命中率明显低于 id 0–29（`TRUTH_EQUIV` 同义词表是按 id 0–29 建的），
>    **很可能需要对 60 例做一次同义词表扩充**，否则规则层会低估准确率。
> 3. 不确定 case 已 dump 到 `results/variants_uncertain.json`（含每个 case 的 truth + 各方法 pred）。

---

## 5. 接手后要做什么（明确任务清单）

### A. 先补全 60 例的最终确定性数字（机制归因前置）

1. 复核 `results/variants_uncertain.json` 里的「规则不确定」case（规则 + 人工，口径见 §2 发现3）；
   或改跑 `deterministic_judge.py` 的 `judge()`（规则层 + DeepSeek 双判兜底）把不确定 case 判完。
2. 若发现 `TRUTH_EQUIV` 对 id 30–59 覆盖不足，扩充 `src/deterministic_judge.py` 的同义词表后重跑。
3. 得到 60 例最终准确率后，回答两个问题：
   - 完整版 73.3%（30 例）在 60 例下是否稳定？refine_cv_qwen 是否仍显著 > refine？
   - 单组件「都不显著、协同才有效」的结论在 60 例下是否成立？

### B. F3 多模型表征（最有论文价值，GLM key 已就绪）

- 跑法（`src/run_variants.py`，`build_doctor` 已支持 glm）：
  ```bash
  # glm 生成 + glm 核对（同模型，验证 GLM 家族内同模型核对是否同样有效）
  python3 src/run_variants.py refine_cv_glm --doctor glm --start 0 --end 30
  # qwen 生成 + glm 核对（跨模型，验证 F3 是否从 qwen↔DeepSeek 泛化到 qwen↔glm）
  python3 src/run_variants.py refine_cv_glm --doctor qwen-plus --start 0 --end 30
  # deepseek 生成 + glm 核对（第三组跨模型）
  python3 src/run_variants.py refine_cv_glm --doctor deepseek --start 0 --end 30
  ```
- 目标：证明/证伪「跨模型语义不对齐」是否 qwen↔DeepSeek 特有，还是跨家族普遍规律。

### C. 机制归因（P2，若协同效应成立）

- 深挖「为什么单一约束有害、完整闭环有效」，可能指向可命名的机制：
  **交互式诊断的证据锚定需要「核对—缺口—反证」闭环，任何单一约束都引入新失败模式**。
- 入口脚本：`src/compare_variants.py`（对比）、`src/collect_failure_metrics.py`、`src/analyze_failures.py`（失效拆解）。

---

## 6. 代码地图（关键文件）

```
src/
  run_variants.py          # 通用跑批：python3 src/run_variants.py <mode> [--doctor] [--start] [--end]
  deterministic_judge.py   # 确定性判定：rule_judge(规则层) + judge(规则+DeepSeek双判兜底)
  compare_variants.py      # 对比各方法准确率 + dump 不确定 case（已修复 load 只读 record/）
  collect_failure_metrics.py / analyze_failures.py  # 失效拆解管线
  llm_client.py            # 多模型 OpenAI 兼容客户端（qwen/deepseek/glm）
  agentclinic_env.py / runner.py / methods.py       # 早期 DDXPlus/AgentClinic 实验

benchmarks/eid-benchmark/src/eid/scenarios/
  refine.py                # 基线（无插件）
  refine_cv.py             # CEV-GR 完整（VERIFIER_PROVIDER="deepseek"，跨模型）
  refine_cv_qwen.py        # 同模型核对（VERIFIER_PROVIDER="qwen"）
  refine_cv_ablations.py   # check/gap/refute 三消融 + qwenmax + GLM 场景（RefineCVGLMScenario）
  meddxagent.py / clinicalagents.py / roleplay.py   # 被移植的方法
  __init__.py / cli.py     # mode 注册（AVAILABLE_MODES）

benchmarks/eid-benchmark/src/eid/agents/chat_agent.py  # 含 DummyTokenCounter（绕过 tiktoken SSL 下载，勿删）
```

---

## 7. 复现手册

### 环境
- Python 3.11（pyenv `/root/.pyenv/versions/3.11.1`）
- 依赖：`camel`、`openai`、`python-dotenv`、`tqdm`（见 `benchmarks/eid-benchmark/pyproject.toml`）
- 已知坑：tiktoken 会因沙箱 SSL 拉不到 tokenizer，已用 `DummyTokenCounter` 绕过；`mcp<2`、`typing_extensions --ignore-installed`。

### 密钥（不入库）
```bash
cp config/api_keys.env.example config/api_keys.env   # 填入 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / ZHIPU_API_KEY
```
- 千问（DashScope，`sk-` / `sk-ws-` 格式）、DeepSeek（`sk-`）、GLM（`id.secret` 格式，`ZHIPU_API_KEY`）。
- `benchmarks/eid-benchmark/.env` 里也需 `OPENAI_API_BASE_URL` + `OPENAI_API_KEY`（跑 eid-eval CLI 时用）。

### 关键命令
```bash
cd /workspace/medical_llm_exp
python3 src/run_variants.py refine_cv_qwen --start 0 --end 30   # 跑 30 例
python3 src/compare_variants.py                                  # 规则层对比 + dump 不确定
```

---

## 8. 数据与结果位置（哪些在库、哪些需重下）

| 内容 | 位置 | 是否入库 |
|------|------|:---:|
| AgentClinic MedQA 200 例（分段） | `benchmarks/eid-benchmark/data/agentclinic_medqa_segmented.jsonl` | ✅ |
| 各方法 trace（record） | `benchmarks/eid-benchmark/results/medqa/<mode>/<model>/10_turns/record/` | ✅ |
| 汇总结果（根目录） | `results/*.json`（失效分析、variants_uncertain 等） | ✅ |
| DDXPlus 原始 CSV | `data/train.csv` / `test.csv` / `validate.csv`（808MB） | ❌ 需重下（hf-mirror: `kasib00gla/ddxplus`） |
| MEDDxAgent 预计算 embeddings | `benchmarks/meddxagent/**/embeddings/`（~67MB） | ❌ 需自生成/重下 |
| AgentClinic 论文 PDF | `benchmarks/agentclinic/media/` | ❌ |

### fork 来源（代码的 upstream，供追溯）
- EID-Benchmark：`https://github.com/NanshineLoong/EID-Benchmark.git`
- AgentClinic：`https://github.com/samuelschmidgall/agentclinic.git`
- MEDDxAgent：`https://github.com/nec-research/meddxagent.git`
- ClinicalAgents：`https://github.com/ZhuohanGe/ClinicalAgents-Code.git`
- MedExAgent：`https://github.com/EndlessCG/medexagent.git`

---

## 9. 已知坑 / 备忘

1. `result_new.json` = ExactMatch 占位，会被 `run_variants --start 30 --end 60` 覆盖；**真正数字看 `compare_variants.py`**（读 `record/`）。
2. 跨模型跑批里 DeepSeek 是每例多次调用，单方法 30 例约 20 分钟，60 例更久；串行稳妥，别开大并发（国产 API 限流）。
3. `compare_variants.py` 的 `load()` 原本会误读方法目录下的 `result.json`，已修复为只读 `record/` 并容错。
4. 同义词表 `TRUTH_EQUIV` 是按 id 0–29 建的，扩样本后需检查 id 30–59 覆盖。
5. GLM 端点：`https://open.bigmodel.cn/api/paas/v4`（OpenAI 兼容），模型 `glm-4-flash`。
