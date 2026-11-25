# P1 措辞对齐（医生查表 + 规范目录）验证 —— 重要负结果

## 结论一句话

「医生查表说规范词」**净 -1.7pp**（56.7% → 55.0%，不显著）。措辞对齐本身有效（修好 8 例），但「公开全量检查目录」诱导医生多问面板外检查、带偏问诊方向（改坏 9 例 + `Not specified` 反升 160 次），收益被抵消。

## 三路对照（同为 qwen-plus、10_turns、60 例、确定性 rule_judge，None 算错）

| 方法 | 准确率 | vs 基线 |
|---|---:|---:|
| closed-world 基线（稀疏表） | 53.3% (32/60) | — |
| 完整面板（对齐前，检查名混乱） | 56.7% (34/60) | +3.4pp |
| **目录对齐（查表）** | **55.0% (33/60)** | **+1.7pp** |

分段：目录对齐旧 30 例 70.0%（-3.3pp vs 面板）、新 30 例 40.0%（持平）。

## 关键证据：公开目录诱导「多问 + 跑偏」

| 指标 | 面板（对齐前） | 目录（查表） |
|---|---:|---:|
| 医生 [TEST] 请求总次数 | 307 | **357（+50）** |
| 平均每例 TEST 次数 | 5.1 | **6.0** |
| 其中返 Not specified | 103 (34%) | **138 (39%)** |
| 全局 'Not specified' 字样 | 492 | **652（反升）** |

「查表」让医生看到 657 项检查清单，于是问了更多检查；但这些检查里大量是**面板没覆盖**的（面板围绕 ground truth 构建，不可能覆盖全部 657 项），反而制造更多假失败。

## 逐例 diff：修好 8 / 改坏 9

**✅ 修好（措辞对齐的真实收益）**——医生查表后问对了关键检查、verbatim 命中：
- medqa_2 Hirschsprung、medqa_24 Meigs 综合征、medqa_33 主动脉缩窄、medqa_43 肠系膜缺血、medqa_45 玫瑰疹、medqa_15 表皮样囊肿、medqa_16 C. diff 结肠炎、medqa_12 髋脱位

**❌ 改坏（目录诱导跑偏）**——医生从目录里挑了不相关检查、被带偏：
- medqa_1 PML → Vitamin E 缺乏、medqa_9 子宫内膜炎 → 产后甲状腺炎、medqa_35 Glanzmann → storage pool 病、medqa_39 原发性血小板增多 → 缺铁、medqa_59 PID → 信息不足、medqa_21 反应性关节炎 → 脓毒性关节炎、medqa_23 IBS → 显微镜结肠炎、medqa_14 屈肌腱断裂 → 痛风、medqa_18 子宫破裂 → 胎盘早剥

## 核心洞察：措辞对齐 与 公开目录 必须拆开

1. **措辞对齐（面板 test 名规范化 + 规范目录）本身是有价值的**——修好 8 例"问对但词对不上"的 case，证明「同检查不同词」是真实损耗。
2. **但「把目录公开给医生、让医生查表」是有害的**——在这个「答案导向、面板稀疏」的 benchmark 上，公开检查目录 = 给医生一张"诱导跑偏的地图"。这与此前方案 B（availability 泄露）的教训一致：**给 agent 暴露"检查空间"信息，是双刃剑**。
3. 净效果为负，`p=1.00` 不显著。

## 正确方向：把措辞对齐「下沉到 reporter 端」，而非「要求医生查表」

- **放弃**：医生查表（公开目录 → 诱导跑偏）。
- **改用**：reporter 端语义归一化——医生自由说词（不被目录带偏），reporter 拿到请求后用规范目录 + token 归一化（`build_exam_catalog` 已产出的 `canonicalize` 逻辑）匹配面板规范名。既保留"措辞对齐"收益，又不诱导跑偏。
- 这与之前方案 A（reporter 语义检索）的区别：现在有了**规范目录 + 映射表**做确定性归一化（而非 LLM 自由裁量"查不查得到"），假命中/假阴性风险大幅降低。

## 产物清单

- `src/build_exam_catalog.py`：从 1387 个脏检查名 → 657 个规范条目（目录 + 脏名映射表）
- `src/build_canonical_panel.py`：面板 test 名对齐到规范名（2355 项 mapped 2354）
- `src/p0_catalog_compare.py`：三路对比脚本
- `manager.py`：`DOCTOR_EXAM_CATALOG` 环境变量注入目录（默认关闭，不影响其他场景）
- `data/exam_catalog.json` + `exam_catalog_map.json`：规范目录 + 映射表
- `data/exam_panels_canonical/` + `agentclinic_medqa_panel_canonical.jsonl`：canonical 面板与数据集
- `refine_cv_qwen_catalog/`：60 例 trace

## 下一步建议

1. **reporter 端语义归一化**（推荐）：医生自由说词，reporter 用 `canonicalize` 归一化匹配 canonical 面板。预期：保留 +8 例的措辞对齐收益，去掉 -9 例的跑偏损失。
2. 目录精简（657 项含同义重复，如 `X` vs `X Measurement`），降低 prompt 成本与噪音。
