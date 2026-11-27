# 过程归因：为什么「没采到关键检查」——真相是 benchmark 环境瓶颈，不是模型知识缺口

## 背景与自我修正

上一轮分析（`missed-key-exam-qualitative-analysis.md`）把「没采到的关键检查」判为
「91.8% 是 standard=教科书该想到却漏了=知识缺口」，这是 **hindsight bias（结果归因）**：
站在正确答案事后反推「该查什么」，却无视了模型当时的认知状态。

用户指出：**模型当时做的是「当时的最优解」**——它可能正在用更优先的检查去分辨更可疑的
假设，关键检查只是排在后面/被环境挡掉。本文改为**过程归因**：回看每个 case 里 doctor 每一轮
`[THOUGHT]`（可观测的决策理由），判断它当时为什么没采到那条检查。

## 方法

对 244 条「没采到的判别性关键检查」，逐条给 DeepSeek 输入「正确诊断 + 该检查 + 该 case 的
doctor 完整 [THOUGHT] 轨迹」，要求**只依据轨迹文本**（不许事后脑补）分类机制：

| 机制 | 含义 | 性质 |
|---|---|---|
| env_unavailable | 请求了该检查，环境返回 unavailable/Not specified | 非模型缺陷 |
| redundant_closed | 已用其他证据闭合诊断，判断此检查冗余 | 非缺陷（当时最优） |
| turn_exhausted | 想到该方向，但排在后面、16 轮用完 | 真问题（采集规划） |
| wrong_priority | 有更高优先假设，正确方向被压 | 真问题（决策取舍） |
| direction_not_considered | 正确诊断方向从未出现在推理里 | 真问题（知识缺口） |

## 结果 1：机制分布（244 条判别性检查）

| 机制 | 条数 | 占比 |
|---|---|---|
| env_unavailable | 151 | **61.9%** |
| turn_exhausted | 40 | 16.4% |
| direction_not_considered | 23 | **9.4%** |
| redundant_closed | 22 | 9.0% |
| wrong_priority | 6 | 2.5% |
| other | 2 | 0.8% |

**结论：非缺陷占 70.9%（61.9% 环境不可用 + 9.0% 理性冗余），真正的「知识缺口」只有 9.4%。**
上一轮「91.8% 是知识缺口」被彻底推翻。

按 case 对错交叉：`direction_not_considered` 在错误 case 占 14.5%、正确 case 占 4.2%
（错误组更高，说明方向没想到确实是答错的贡献之一，但体量很小）。

## 结果 2：硬证据——医生的检查请求 72.6% 落空

直接统计方案A 200 例里医生的 `[TEST]` 请求与 reporter 返回：

| | 次数 | 占比 |
|---|---|---|
| TEST 请求总数 | 1246 | — |
| 命中（返回结果） | 342 | 27.4% |
| **未命中（Not specified / unavailable）** | **904** | **72.6%** |

**医生每 4 次检查请求，只有 1 次能拿到结果。** 这不是医生不想查、不会查，而是检查请求
被 reporter 大量挡掉。

## 根因：closed-world reporter + 检查面板稀疏 + 面板补全未完成

代码（`eid/prompts/manager.py`）里 `REPORTER_MODE = "closed"`，且注释记录了此前已知的结论：

> 查不到（真正冷门检查）才返回 'Not specified'…… 面板补全后"查得到"成为常态。

即：reporter 是**逐字检索** exam_facts 里的检查项，医生请求的检查只要不逐字命中就返回
"Not specified"。而当前实验数据 `agentclinic_medqa_segmented.jsonl` 的 exam_facts 均值仅
**14.87 条/例**，非常稀疏；「离线补全完整检查面板」的方案（`data/exam_panels` +
`build_panel_dataset.py`）**只完成了 61/200 例**，且 refine_cv_cover 实验跑的是**未补全的原始数据**。

这条链把之前的「采集损失 15pp」重新归因：
> 不是「医生不会采关键检查」，而是「关键检查根本不在面板里 / reporter 不给结果」。

## 修正后的主战场判断

| 之前的判断 | 修正后 |
|---|---|
| 主战场=关键证据定向采集，gap 是知识缺口(91.8%) | gap 主因是 **benchmark 环境瓶颈**（72.6% 检查落空），知识缺口仅 9.4% |
| 下一步=注入「症状→检查」临床路径知识 | 下一步=**完成检查面板补全（61→200），消除人为环境约束** |

真正值得做的顺序：
1. **完成检查面板补全**（`data/exam_panels` 61→200 例，`build_panel_dataset.py` 生成
   `agentclinic_medqa_panel.jsonl`），用补全数据重跑方案A，看关键检查覆盖率能否从 61.1% 大幅回升。
2. 若补全后覆盖率仍不足，再回头处理 `turn_exhausted`（16.4%）与 `direction_not_considered`（9.4%）。

## 数据文件

- 过程归因：`results/coverage_analysis/process_attribution.json`（244 条，含机制+证据）
- 汇总：`results/coverage_analysis/process_attribution_summary.json`
- 脚本：`src/process_attribute_missed_exams.py`（归因）、`src/summarize_process_attribution.py`（汇总）
