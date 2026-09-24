"""OpenRouter adapter; model and prompt are explicit, independently configurable inputs."""
import json
import os
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env.local", override=False)

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "accountability_system.txt"


def list_models():
    with urlopen("https://openrouter.ai/api/v1/models", timeout=15) as response:
        payload = json.load(response)
    return sorted({m["id"] for m in payload["data"] if "text" in m.get("architecture", {}).get("output_modalities", ["text"])})


def generate_analysis(summary, model, system_prompt, api_key=None, client=None):
    if not model.strip() or not system_prompt.strip():
        raise ValueError("Choose a model and provide a system prompt.")
    if client is None:
        from openai import OpenAI
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY to enable AI analysis.")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=60, max_retries=1)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system_prompt},
                               {"role": "user", "content": json.dumps(summary, allow_nan=False)}],
        max_tokens=2500,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The model returned no analysis. Try another model.")
    return content
