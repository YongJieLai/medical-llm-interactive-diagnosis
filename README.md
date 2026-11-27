# 医学交互诊断 LLM：失效场景分析 → 零训练插件解决方案

面向「医疗大模型交互式问诊 / 诊断」的研究实验库。目标不是刷 SOTA，而是走通一条完整链路：

> **找到一个失效场景 → 逐层归因到真正的失效环节 → 提出一个零训练插件 → 用对照实验证明它解决了问题、且能迁移到其它方法底座上。**

判分口径统一：`deterministic_judge`（答案归一化后与 ground_truth 严格匹配），AgentClinic-MedQA **200 例 × 16 turns × qwen-plus**（全角色同模型）。

## 研究动机（Motivation）

### 1. 研究对象：交互式诊断（在线证据采集）方法

我们研究的是**交互式诊断**（interactive diagnosis）方法——让 LLM 扮演医生，在对话中**逐轮主动提问、在线采集患者的病史与检查结果**，最后给出诊断。这类方法的核心动作是"在线证据采集"（online evidence collection），与"把病例全部信息一次喂给模型做单次 CoT"（静态上界 cot）形成对照。

研究起点是这类方法的三个上游基线（EID-Benchmark 原版，200 例 × 16 轮 × qwen-plus 同模型）：

| 方法 | 机制 | 准确率 | 信息采集覆盖率 |
|---|---|---|---|
| roleplay | 自由问诊 | 49.0% | 32.3% |
| refine | 问诊 + 迭代门控 | 55.0% | 33.2% |
| sc | 采集 + 总结 | 59.0% | 32.4% |

### 2. 共性问题：失效在采集层

三个方法有同一个显著失效：准确率比静态全信息上界（**85.5%**）低 **26–36pp**，且**信息采集覆盖率都只有 ~33%**。

为把落差定位到具体环节，做了 **oracle 上界实验**——固定同一个 qwen-plus 诊断器、同一个 prompt，唯一变量是喂给它的证据：

| 证据档 | 准确率 |
|---|---|
| oracle_full（喂完整证据） | **87.5%** |
| 交互式采集（refine 基线） | **55.0%** |

**诊断器拿到完整证据能到 87.5%，交互式采集却只有 55%** → 落差不在诊断器、而在采集层：这些方法的采集器**采不到判别性证据**（采到的是共性/噪音证据）。

### 3. 我们的插件：双组件、零训练、纯 prompt

针对"采集层"这个共性问题，在上游方法上挂一个双组件插件：

| 组件 | 层 | 做法 |
|---|---|---|
| 8 维证据框架 | 采集层 | 给 doctor 一份静态证据类别清单，锚点 = 证据类别缺口（不是诊断假设优先级） |
| CEV-GR 验证环 | 诊断层 | claim / gap / refutation / decisive 四检查，消化广覆盖证据 |

两个组件**可加、零训练、纯 prompt 级**。

### 4. 结果：提升多少点、指标怎么变

**refine 底座的完整消融链**：

| 方法 | 准确率 | 信息覆盖率 |
|---|---|---|
| refine（起点） | 55.0% | 33.2% |
| +8 维框架（采集层） | 60.0% | **48.5%** |
| +CEV-GR（诊断层）= 方案A | **65.0%** | **50.1%** |

**roleplay 底座**：

| 方法 | 准确率 |
|---|---|
| roleplay（起点） | 49.0% |
| +8 维框架 | 53.5% |
| +CEV-GR = roleplay_cover | **64.0%** |

**跨底座汇总**：

| 底座 | 原样 | +插件 | Δ |
|---|---|---|---|
| refine | 55.0% | 65.0% | **+10.0pp** |
| roleplay | 49.0% | 64.0% | **+15.0pp** |

**两个指标的改变**：
1. **准确率**：refine +10.0pp、roleplay +15.0pp；
2. **信息采集覆盖率**：~33% → ~50%（+17pp），且这个提升**全部来自采集层组件**（8 维框架）；诊断层组件（CEV-GR）不改覆盖率、只把采到的证据转化成准确率——两组件各司其职，印证"问题在采集层、插件修在采集层"。

### 5. 边界（诚实）

| 底座 | 原样 | 完整插件 | Δ | 判定 |
|---|---|---|---|---|
| **refine**（采集-诊断解耦 + 验证环） | 55.0% | **65.0%** | +10.0 | ✅ |
| **roleplay**（采集-诊断解耦） | 49.0% | **64.0%** | +15.0 | ✅ |
| sc（summarizer 一步诊断，无验证环） | 59.0% | — | 8维框架单独 −2.5 | ⚠️ 依赖验证环 |
| react（采集-诊断单 agent 耦合） | 53.0% | 50.0% | −3.0 | ❌ 结构不兼容 |

**三条机制结论**：① 8 维框架是底座无关的采集组件（refine +5.0 / roleplay +4.5）；② 采集组件的收益**依赖诊断层验证环来消化**（sc 反例）；③ "采集-诊断耦合"是硬边界（react 反例）。另有 6 个负结果锁定的规律：**采集层任何"动态优先级/选择"措辞都会退化成"少问快结束"，唯一有效的是"静态完整清单"**。

## 关键数字（200 例口径）

| 方法 | 机制 | 准确率 |
|---|---|---|
| cot | 静态全信息（上界） | 85.5% |
| **refine_cv_cover（方案A = 完整插件）** | 8维框架 + CEV-GR | **65.0%** |
| roleplay_cover | 方案A 移植到 roleplay 底座 | 64.0% |
| refine_cover | 消融：8维框架（无 CEV-GR） | 60.0% |
| sc | SC 基线 | 59.0% |
| refine_cv | CEV-GR 跨模型四检查 | 58.5% |
| refine_cv_freebudget | 消融：只去效率、不加框架 | 57.5% |
| refine_cv_qwen | CEV-GR 同模型四检查 | 57.5% |
| **refine** | **REFINE 基线（无插件）** | **55.0%** |
| sc_cover | 8维框架（无 CEV-GR，SC 管线） | 56.5% |
| react_cover | 方案A 移植到 react 底座 | 50.0% |
| roleplay | roleplay 基线 | 49.0% |

> 完整 36 方法总账见 [docs/experiments-ledger.md](docs/experiments-ledger.md)。

## 目录结构

```
src/                                          # 实验脚本（跑批 / 判分 / 覆盖分析 / 失效归因）
benchmarks/eid-benchmark/                      # EID-Benchmark fork，所有场景在 src/eid/scenarios/
benchmarks/{agentclinic,meddxagent,...}/       # 被移植方法的原始代码
config/                                        # api_keys.env.example 模板（真实 key 不入库）
data/                                          # 检查面板（200 例）等小规模输入
results/coverage_analysis/                     # 逐例判分 percase_*.json + 覆盖率 + 关键事实标注
docs/                                          # 研究笔记、实验报告（core-narrative / experiments-ledger）
DELIVERY.md                                    # 交付文档（当前第一入口）
```

## 快速开始

```bash
# 1. 配置密钥
cp config/api_keys.env.example config/api_keys.env   # 填 DASHSCOPE_API_KEY

# 2. 跑方案A（完整插件，200 例，6 分片并行）
for i in 0 1 2 3 4 5; do
  SHARD_ID=$i SHARD_TOTAL=6 python3 src/run_paper_modes.py refine_cv_cover 200 16 3 &
done; wait

# 3. 判分
python3 src/analyze_coverage.py refine_cv_cover
```

`src/run_paper_modes.py <mode> [N=200] [max_turns=16] [workers]` 支持全部 36 个 mode（论文 5 基线 + refine_cv 系列 + 外部方法 + 跨底座移植 + 消融），完整列表见 `--help` 或 `experiments-ledger.md`。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/core-narrative.md](docs/core-narrative.md) | **核心主线**：失效分析 → 插件 → 解决（含佐证数据 + 干净/作废标注） |
| [docs/experiments-ledger.md](docs/experiments-ledger.md) | 实验总账：36 方法索引（mode / 代码文件 / 结果文件 / 准确率）+ 消融链 |
| [docs/plugin-portability-cover.md](docs/plugin-portability-cover.md) | 完整插件跨底座移植结果 |
| [docs/discriminating-key-exam-definition.md](docs/discriminating-key-exam-definition.md) | "判别性关键检查"度量标准定义（P0，待重标注） |
| [DELIVERY.md](DELIVERY.md) | 交付文档 |

## 接手

先读 **[docs/core-narrative.md](docs/core-narrative.md)**（主线）和 **[docs/experiments-ledger.md](docs/experiments-ledger.md)**（总账）。

> ⚠️ 诚实标注：oracle_key 80%、「关键检查覆盖率」42.3%/61.1%/59.6% 等数字**依赖旧 confirmatory 定义，已作废**，待按判别性定义重标注后重算（见 core-narrative.md §3.4、§7）。
>
> ⚠️ 诚实标注（cot 上界口径修正）：本表 cot 静态上界为 **85.5%（171/200）**。早期曾出现 94.0%（188/200），系 `analyze_coverage.py` 把 cot 的 `THOUGHT+DIAGNOSIS` **全文**喂给 `deterministic_judge` 判分——规则层"truth 是 pred 子串"命中 CoT 思考过程中提到的正确病名即判对（如模型 THOUGHT 里讨论过热射病、最终却写"神经阻滞剂恶性综合征"，也被算对）。正确口径是 `summarize_paper_modes.py` 先 `extract_cot_diagnosis` 提取 `DIAGNOSIS:` 后的最终诊断再判分。94.0% 已作废。
