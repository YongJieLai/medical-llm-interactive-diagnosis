# gap 问错深度分析：类型②④⑤ + "做不了的检查"到底是不是医生的错

> 分析对象：`refine_cv_qwen_gap`（gap-only 消融，60 例完整）的 verifier `gaps` 反馈
> 分析日期：2026-11-24

---

## 0. 一句话结论

1. **类型②④⑤（约 7 个 case）才是 gap 机制真正该修的**——修 verifier 的判断质量，不透题。
2. **类型①"问不到的检查"（71%）几乎全是 benchmark 封闭世界 artifact，不是医生问错**：这些检查真实医院都做得起、做了会显示正常/阴性（合理排除性检查），而不是"医院根本做不了"。
3. **根因是我们自己的 fork 改坏了判定口径**：上游 EID-Benchmark / 原版 AgentClinic 都是"查不到 → 返回 Normal"，我们改成了"查不到 → 返回 Not specified"，人为制造了这 71% 的假失败。

---

## 1. "做不了的检查"：其他 benchmark 是怎么做的（关键证据）

三家（其实是四家）对"医生问了一个不在数据里的检查"的处理方式，正是分歧点：

| 来源 | 判定口径 | 原文 | 等价语义 |
|------|---------|------|---------|
| **原版 AgentClinic** | 开放式（open-world） | `agentclinic.py` MeasurementAgent：*"If the requested results are not in your data then you can respond with NORMAL READINGS."* | 查不到 → **返回正常读数**（= 做了显示正常/阴性） |
| **上游 EID-Benchmark** | 开放式（open-world） | `NanshineLoong/EID-Benchmark` `manager.py`：*"If the requested test results are not found in the list, assume the finding is non-significant and return 'Normal'."* | 查不到 → **假设非显著性，返回 Normal** |
| **我们的 fork（当前）** | 封闭式（closed-world） | 当前 `manager.py`：*"MUST return 'Not specified'… NEVER report a missing result as 'Normal'… Only report 'Normal' when the list EXPLICITLY states a normal/negative result."* | 查不到 → **Not specified（没信息/死路）** |
| **EGIP（2025 新协议）** | 显式不可用 | *"若结果缺失或未知，就保持为不可用，而不是用生成的发现取代"* + *"未知结果也不会被当作正常"* + 提供可请求检查目录与 availability 元数据 | 查不到 → **显式"不可用" + 告诉医生哪些检查可用** |

**这就直接回答了用户的问题**：

- "做了以后会显示正常、所以没信息" → 这正是**原版 AgentClinic / 上游 EID-Benchmark 的 open-world 口径**（返回 Normal）。在这种情况下，医生问这个检查是**合理的**（排除性检查，做了阴性即排除）。
- "医院根本做不了这个检查" → 这才是**不该问**的；EGIP 用 availability 元数据显式解决这一档（告诉医生"这个检查不存在/不可用"）。

**我们的 fork 把两档混成了同一个 "Not specified"，于是把"合理排除性检查"错判成"问错"，这就是 71% 假失败的来源。**

### 为什么上游用 open-world（返回 Normal）？

因为 benchmark 的事实表只收录了"能指向正确答案"的最小事实集（通常是 1 个阳性检查）。真实医生会做一串排除性检查（B12、TSH、CBC、常规影像……），这些做完都是阴性。open-world 口径让这些检查返回 Normal，医生能高效地排除、收敛到正确答案；closed-world 口径让它们返回 Not specified，医生被卡死在"每个合理检查都查不到"的死循环里。

### open-world 的代价（为什么不能无脑改回去）

返回 Normal 有风险：如果医生问对了检查但**措辞与事实表 key 不一致**（如问 "JC virus PCR" 而事实表存的是 "MRI: lesions consistent with PML"），open-world 会误返 Normal → 假阴性 → 医生错误排除真病。这正是 EGIP 用"显式 availability 元数据"解决的点：让医生知道哪些检查是可请求的，而不是把"不可用"伪装成"正常"。

---

## 2. 类型①的定性：做不了 vs 做了显示正常

数据：`gap_only`（60 例）中干净的 "Not specified" 检查请求 **172 条**（53 个 case）。

| 类别 | 数量 | 占比 | 定性 |
|------|:---:|:---:|------|
| **(b) 标准检查·做了会显示正常/阴性** | 164 | 95.3% | 医生问得合理，benchmark 应返回 Normal |
| (a) 非检查类请求 | 5 | 2.9% | 免疫接种记录、3天膀胱日记、身高体重——不是"检查"，reporter 无结果可返 |
| ④ 方向错（问错病的检查） | 3 | 1.7% | CF 测试问 Hirschsprung、霍奇金标记问 DLBCL 等——这才是 gap 该修的 |

**结论：172 条里几乎没有"医院根本做不了"的检查。** 所谓"做不了的检查"这一档在我们的数据里基本是空的。绝大多数（95%）是 B12、维生素 E、TSH、CBC、肝功能、MRI/CT/超声、活检、培养、ECG/EEG、裂隙灯、皮肤镜、KOH 涂片、关节穿刺、胸腔穿刺、阴道超声、MRCP、核型、性激素等**真实医院都做得起、做完是阴性**的标准检查。

> 即：类型①的 71% 不是 gap 机制的问错，而是**判定层（reporter 口径）的假阴性**。

---

## 3. 类型②④⑤ 深度分析（这才是 gap 真该修的）

### ② 过度细分亚型：诊断已对，gap 再逼分型（3 case）

| case | truth | gap 在问什么 | 后果 |
|------|-------|-------------|------|
| `medqa_0` 重症肌无力 | "Myasthenia gravis" | "ocular vs generalized MG"、"MGFA Class I vs 早期 generalized"、"ocular vs seronegative NMJ" | 最终 pred 过度具体化为 **"Ocular myasthenia gravis"**（母病已对，被 gap 逼成亚型） |
| `medqa_11` 复杂部分性发作 | "Complex partial seizure" | "mesial temporal vs insular/operculo 致痫灶"、"右海马硬化 vs 颞外"、"FDG-PET 代谢定位" | pred 过度定位为 **"右颞叶非病灶性 MTLE"**（远超 truth 粒度） |
| `medqa_12` 髋关节后脱位 | "Posterior hip dislocation" | 坐骨神经损伤、血管损伤、骨筋膜室综合征（并发症层面） | pred 已对，gap 仍追问并发症 |

**修法**：verifier 的 gap 判定加一道 gate——"该缺口是否会**改变诊断或治疗**？"若母病已成立、仅剩亚型/并发症细分，且细分不改治疗，则不产生 gap。

### ④ 医学问错方向：gap 问的是"别的病"的检查（3 case）

| case | truth | gap 让查什么 | 问题 |
|------|-------|-------------|------|
| `medqa_3` 弥漫大 B 细胞淋巴瘤（DLBCL） | DLBCL | **Reed-Sternberg 细胞 / CD15 vs CD20** | RS 细胞是**霍奇金淋巴瘤**（cHL）标志，truth 是**非霍奇金**（DLBCL）。金标准活检（CD20+ 高增殖）就在事实表里，gap 却把医生引向霍奇金标记 |
| `medqa_13` 先天性巨结肠（Hirschsprung） | Hirschsprung | **汗液氯 / iRT**（囊性纤维化 CF 测试） | CF 测试问的是"别的病"。金标准直肠活检（无神经节细胞）就在事实表里 |
| `medqa_57` 胰腺癌 | Pancreatic adenocarcinoma | **胸片看纵隔/肺门淋巴结**（淋巴瘤/肺癌/结节病） | 追的是肺/淋巴瘤鉴别，与胰腺癌无关 |

**修法**：verifier 产出 gap 时，必须把"缺口信息"锚定到**当前主假设的鉴别诊断树**里——禁止跨疾病（为别的病）要检查。

### ⑤ 问错重点：同病域内问了次要的（1 case）

| case | truth | gap 让查什么 | 问题 |
|------|-------|-------------|------|
| `medqa_17` 坏疽性脓皮病（PG） | Pyoderma gangrenosum | **pathergy 现象**（同形反应，Behçet 特征） | PG 的关键检查是**皮肤活检**（金标准就在事实表："中性粒细胞浸润、无感染，符合 PG"）。gap 把医生引向 pathergy 而非活检 |

**修法**：verifier 的 gap 应优先指向**能直接确诊/排除的检查**（活检、培养、关键影像），而非支持性体征（pathergy）。

---

## 4. 对插件的修正建议（不透题）

1. **① 类（95%）不动 gap，动判定层**：
   - 方案 A（对齐上游）：把 reporter 改回"查不到 → 返回 Normal/阴性"，恢复 open-world；
   - 方案 B（更稳，EGIP 式）：reporter 返回"不可用"时附带 availability 元数据，并在评估口径里把"合理排除性检查的 Not specified"不计入 coverage 损失 / 不计入"烧轮次"。
   - ⚠️ 注意方案 A 的假阴性风险（问对检查但措辞不同 → 误返 Normal），故推荐方案 B 或 A+措辞归一化。

2. **②④⑤ 修 verifier 判断质量（不透题，因为不约束到金标准 facts）**：
   - 加"亚型 gate"（②）：细分不改诊断/治疗则不产生 gap；
   - 加"鉴别诊断树锚定"（④）：gap 必须落在当前主假设的鉴别诊断树内，禁止跨疾病；
   - 加"确诊检查优先"（⑤）：gap 优先指向能确诊/排除的检查，而非支持性体征。

3. **不要做的事**：把 gap 约束到金标准 `atomic_facts`（= 透题），这会毁掉整个验证的意义。

---

## 5. 复现与数据位置

- 提取脚本：`/tmp/extract_qa.py`、`/tmp/clean_ns.py`、`/tmp/extract_gaps.py`
- 干净分类结果：`/tmp/ns_clean.json`（172 条，含 case/truth/tests）
- 上游对照：`/tmp/eid_upstream_manager.py`（第 106 行 `return 'Normal'`）
- 原版对照：`benchmarks/agentclinic/agentclinic.py`（MeasurementAgent.system_prompt，`NORMAL READINGS`）
- 我们当前口径：`benchmarks/eid-benchmark/src/eid/prompts/manager.py`（第 107 行 `Not specified`）
