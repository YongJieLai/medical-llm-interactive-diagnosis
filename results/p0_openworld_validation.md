# P0 reporter 口径修复（方案 A：open-world）验证报告 · 负结果

> 实验：把 reporter "查不到检查" 的口径从 closed-world（返回 `Not specified`）
> 改为 open-world（语义匹配 → 否则返回 `Normal`），对齐上游 AgentClinic / EID-Benchmark。
> 验证集：`refine_cv_qwen` 60 例（qwen-plus，10_turns），确定性 `rule_judge`（None 算错）。

## 一、结果

| 口径 | 正确 | 准确率 |
|---|---:|---:|
| closed-world（基线） | 32/60 | **53.3%** |
| open-world（方案 A） | 24/60 | **40.0%** |

**Δ = -8 例（-13.3pp），方案 A 显著有害。** McNemar `b=16, c=8, p=0.153`（不显著但方向明确为负）。

## 二、口径切换确实生效

| 口径 | `Not specified` | `Normal` |
|---|---:|---:|
| closed-world | 868 次 | 550 次 |
| open-world | 5 次 | 1752 次 |

## 三、失败机制：假阴性

修好 8 例 / **改坏 16 例**。改坏的 16 例几乎全是**假阴性**——医生问对了检查，但
措辞与事实表 key 不一致，reporter 误返 `Normal`，医生据此**错误排除真病**：

| case | truth | closed-world 诊断 | open-world 诊断 |
|---|---|---|---|
| medqa_2 | Hirschsprung disease | ✓ Hirschsprung | ✗ Occult acute otitis media |
| medqa_24 | Meigs syndrome | ✓ Meigs' syndrome | ✗ SLE |
| medqa_29 | Disseminated Gonococcal Infection | ✓ DGI | ✗ Functional wrist pain |
| medqa_3 | DLBCL | ✓ DLBCL | ✗ Insufficient information |
| medqa_16 | C. difficile colitis | ✓ CDI | ✗ non-CD antibiotic-associated |

## 四、结论

1. **方案 A（纯 open-world）不可行**。假阴性（-13.3pp）远大于消除 71% 假失败的收益。
2. 加入的"语义匹配护栏"（先语义匹配、真无关才返 Normal）**不足以阻止假阴性**——
   LLM 没有可靠地区分"同检查不同措辞"与"真无关检查"。
3. **印证了分析文档的预判**：open-world 的假阴性风险是真实且严重的，不能无脑改回去。

## 五、下一步：方案 B（EGIP 式 availability 元数据）

reporter 查不到时**不返回 Normal、也不返回 Not specified**，而是返回
"**不可用 + 可请求检查目录**"，让医生：
1. 知道"这个检查此环境不可用"（不会误排除）；
2. 知道"哪些检查可请求"（能转向正确的检查方向）。

需改动三处：
1. `get_reporter_system_prompt`：查不到 → 返回"Not available in this environment"，
   并附上 exam_facts 中**可请求的检查类别清单**；
2. doctor prompt：引导医生收到"不可用"时转向可用检查；
3. 评估口径：把"因不可用导致的 Not available"不计入 coverage 损失 / 不计入烧轮次。

## 六、代码状态

- `prompts/manager.py` 新增开关 `REPORTER_OPEN_WORLD`（默认 **False**，即 closed-world）。
- 开关保留供方案 B 消融；注释已记录本负结果。
- 验证脚本：`src/p0_openworld_compare.py`。
