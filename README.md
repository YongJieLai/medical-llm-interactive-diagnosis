# Medical LLM Interactive Diagnosis — CEV-GR 失效分析与机制归因

面向「医疗大模型交互式问诊 / 诊断」的研究实验库。目标不是刷 SOTA，而是：

1. 把文献中的主动问诊 / 证据验证方法移植到 AgentClinic（MedQA）场景；
2. 检查这些方法的**失效场景**；
3. 提出并验证一个零训练插件 **CEV-GR**（Cross-model Evidence Verification + Gap-driven Refutation）；
4. 定位一个**我们自己的可复现发现：F3 跨模型语义不对齐（cross-model semantic misalignment）**。

## 一句话结论

> 交互式诊断里，文献共识的「生成与核对解耦（跨模型）」在我们场景下**系统性劣于同模型自我核对**：
> 跨模型核对 40% < 基线 60% < 同模型核对 **73.3%**。根因不是核对模型弱，而是**模型间语义边界不一致**，
> 核对模型把生成模型的「临床推理概括」误判成「幻觉证据」，反复打回直到诊断崩溃。

## 关键数字（30 例，确定性 judge：规则层 + 人工复核）

| 方法 | 机制 | 准确率 |
|------|------|:---:|
| refine | 基线（无插件） | 60.0% |
| refine_cv（DeepSeek 核对 qwen） | 跨模型核对 | 40.0% |
| **refine_cv_qwen（qwen 核对 qwen）** | **同模型核对·完整** | **73.3%** |
| refine_cv_qwen_check / gap / refute | 三组件单消融 | 63.3% / 56.7% / 56.7% |
| refine_cv_qwenmax（qwen-max 核对） | 强跨模型 | 66.7% |
| refine_cv（全 DeepSeek 同模型） | 稳健性 | 76.7% |

## 目录结构

```
src/                       # 实验脚本（跑批 / judge / 对比 / 失效分析）
benchmarks/eid-benchmark/  # EID-Benchmark fork，CEV-GR 场景在此（src/eid/scenarios/）
benchmarks/{agentclinic,meddxagent,clinicalagents,ClinicalAgents-Code,medexagent}/  # 被移植方法的原始代码
config/                    # api_keys.env.example 模板（真实 key 不入库）
data/                      # 小规模输入/标注（DDXPlus 大 CSV 不入库，见 .gitignore）
results/                   # 根目录汇总结果（失效分析等）
docs/                      # 研究笔记、精读、实验报告（md + xlsx）
HANDOFF.md                 # 交接文档（接手请先读这个）
```

## 快速开始（复现 30 例 CEV-GR 对比）

```bash
# 1. 配置密钥
cp config/api_keys.env.example config/api_keys.env   # 填入 qwen / deepseek / glm 任一

# 2. 跑一个变体（30 例）
cd benchmarks/eid-benchmark
python3 ../../src/run_variants.py refine_cv_qwen --start 0 --end 30

# 3. 确定性 judge 对比（读 record trace，输出准确率 + 待人工复核 case）
python3 ../../src/compare_variants.py
```

## 接手

看 **[HANDOFF.md](./HANDOFF.md)**，里面有任务状态、下一步和已沉淀的结论。
