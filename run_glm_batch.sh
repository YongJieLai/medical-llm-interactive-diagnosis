#!/bin/bash
# 用智谱 GLM (glm-4-flash) 跑 AgentClinic 四方法对齐实验
# 与之前 qwen-plus 完全同配置：medqa 前15例 / 10 turns / 串行
set -u
cd /workspace/medical_llm_exp/benchmarks/eid-benchmark

# 从配置读取密钥（ZHIPU_API_KEY 已写入）
set -a
source /workspace/medical_llm_exp/config/api_keys.env
set +a

export PYTHONPATH=src
export OPENAI_API_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export OPENAI_API_KEY="$ZHIPU_API_KEY"

python3 -m eid.cli \
  --datasets medqa \
  --modes roleplay refine meddxagent clinicalagents \
  --doctor-model glm-4-flash \
  --max-items 15 \
  --max-turns 10 \
  --max-workers 1 \
  --output-dir results
