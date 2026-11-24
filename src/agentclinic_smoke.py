"""AgentClinic 环境最小冒烟：1 病例 × vanilla 医生，验证 LLM 患者/测量/医生链路。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_llm
from src.agentclinic_env import load_medqa_cases, run_vanilla, is_correct

client, model = get_llm("qwen")
print(f"provider=qwen model={model}")

cases = load_medqa_cases("medqa", limit=1)
case = cases[0]
print(f"病例: {case['case_id']} | 真值: {case['diagnosis']}")
print(f"主诉: {case['objective']}")
print("-" * 60)

result = run_vanilla(client, model, case, max_turns=12, verbose=True)

print("-" * 60)
print(f"模型诊断: {result['diagnosis']!r}")
print(f"正确答案: {result['truth']!r}")
print(f"判对(规则): {is_correct(result['diagnosis'], result['truth'])}")
print(f"轮数: {result['turns']}")
print(f"动作分布: ", {a: sum(1 for t in result['trace'] if t['action'] == a)
                     for a in set(t['action'] for t in result['trace'])})
