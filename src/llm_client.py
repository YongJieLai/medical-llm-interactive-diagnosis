"""多模型 OpenAI 兼容客户端（千问 / DeepSeek / GLM）。

API key 通过环境变量注入：
- DASHSCOPE_API_KEY  千问（阿里云百炼）
- DEEPSEEK_API_KEY   DeepSeek
- ZHIPU_API_KEY      智谱 GLM

模型名可用环境变量覆盖（默认便宜档）：
- QWEN_MODEL     默认 qwen-plus
- DEEPSEEK_MODEL 默认 deepseek-chat
- GLM_MODEL      默认 glm-4-flash
"""
import os
import json
from pathlib import Path
from openai import OpenAI


def _load_env_file():
    """从 config/api_keys.env 读取 key（不覆盖已存在的环境变量）。"""
    p = Path(__file__).resolve().parent.parent / "config" / "api_keys.env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env_file()

PROVIDERS = {
    "qwen": {
        "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key_env": "DASHSCOPE_API_KEY",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "model_env": "GLM_MODEL",
        "default_model": "glm-4-flash",
    },
}


def get_llm(provider: str):
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"缺少环境变量 {cfg['api_key_env']}（{provider} 的 API key）")
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
    model = os.getenv(cfg["model_env"], cfg["default_model"])
    return client, model


def chat(client, model, messages, temperature=0.0, max_tokens=2048, json_mode=False):
    """统一封装，返回文本（json_mode 时返回解析后的 dict 或原文本）。"""
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    if json_mode:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def available_providers():
    return [p for p in PROVIDERS if os.getenv(PROVIDERS[p]["api_key_env"], "")]
