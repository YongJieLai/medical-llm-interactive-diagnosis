# REFINE (EID-Benchmark) 千问接入验证报告

## 一、结论（一句话）

**一个千问 API key 就能把 REFINE 全套跑起来，链路已打通；成本可控，瓶颈在"调用量"而非"API 数量"。**

## 二、环境搭建（已解决的关键问题）

| 问题 | 解决 |
|---|---|
| camel-ai 未安装 | pip 装到 pyenv 3.11（camel-ai 0.2.90） |
| mcp 版本冲突（FastMCP 缺失） | 降级 `mcp<2`（mcp 1.30.0） |
| tiktoken 下载编码文件 SSL/网络失败 | 注入 `DummyTokenCounter` 绕过（改 `chat_agent.py`），不再依赖 OpenAI tokenizer |
| 千问 base_url | `.env` 配 `https://dashscope.aliyuncs.com/compatible-mode/v1` |

改造点：`src/eid/agents/chat_agent.py` 新增 `DummyTokenCounter`（近似 token 计数，仅满足 CAMEL 记账），避免第三方端点被 OpenAI tokenizer 卡死。

## 三、实测调用量与耗时（千问 qwen-plus，max-turns=6）

| 模式 | 单病例 LLM 调用 | 单病例耗时 | 说明 |
|---|---|---|---|
| **roleplay**（下界） | ~10 次（含 judge） | ~14 s | doctor+patient+reporter 三 agent |
| **refine**（论文方法） | ~15 次（含 judge） | ~64 s | 另加 summarizer+diagnostician+verifier 三角色 |

refine 单病例角色分布示例（medqa_0，重症肌无力，诊断正确）：

```
doctor 5 + patient 1 + reporter 3 + summarizer 1 + diagnostician 1 + verifier 1 = 12 次 + judge 1 = 13 次
```

## 四、refine 小样本结果（2 病例）

| 病例 | 诊断 | 真值 | 结果 |
|---|---|---|---|
| medqa_0 | Myasthenia gravis | Myasthenia gravis | ✓ |
| medqa_1 | Vitamin E deficiency neuropathy | PML（进行性多灶性脑白质病） | ✗ |

信息覆盖率约 0.21（6 轮内只挖到约 1/5 信息）——说明交互式证据获取确实是瓶颈，即使诊断对，过程也大量漏挖。

## 五、资源评估：几个 API 够不够

**数量上：1 个就够。** 所有角色（医生/患者/报告者/总结者/诊断者/验证者/裁判）都走同一个 OpenAI 兼容端点，只是 `model` 名不同，不需要多账号。

**真正的约束是调用量：**

| 规模 | 估算调用量 |
|---|---|
| 3-5 病例 × roleplay+refine | ~75 次（已验证可行） |
| 200 病例 × 5 模式（cot/roleplay/react/sc/refine） | ~8k-10k 次（max-turns=6） |
| 同上，max-turns=16（论文默认） | ~15k-20k 次 |

**建议：**
1. **起步 1 个千问 key 足够**，成本约几元到几十元级别（qwen-plus 单次调用 token 量很小）。
2. **限流是主要风险**：默认 `--max-workers 1` 最稳但慢；若要提速，用多个 key（千问/DeepSeek/GLM）分散并发，或把不同角色挂到不同 key 上。
3. **跨模型对比**（你手上 4 家 key）才是多 key 的真正价值——同一方法换 backbone 看失效差异，而不是为了加速。

## 六、下一步建议

1. 先按 **roleplay（下界）+ refine（方法）** 两条，各跑 10-20 病例，采集失效场景（当前 2 病例里已看到 PML 被错诊成维生素 E 缺乏）。
2. 若调用量扛得住，再补 react / sc 两档，凑齐"下界 → 方法 → 方法+验证"的完整对照。
3. 失效分析重点看：**信息覆盖率低、开检查决策、verifier 是否真能拦下错误诊断**（medqa_1 的 verifier 就没拦住 PML 错诊）。
