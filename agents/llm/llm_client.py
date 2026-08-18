"""Create OpenAI-compatible clients from configuration."""

import os

from openai import OpenAI


def create_llm_client(args):
    provider = getattr(args, "llm_provider", "openai").lower()
    settings = {
        "openai": ("OPENAI_API_KEY", None),
        "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    }
    if provider not in settings:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    env_name, base_url = settings[provider]
    api_key = os.environ.get(env_name) or getattr(args, "api_key", None)
    if not api_key:
        raise ValueError(f"Set {env_name} or Trainer.api_key.")

    options = {"api_key": api_key}
    if base_url:
        options["base_url"] = base_url
    if provider == "openrouter":
        headers = {
            "HTTP-Referer": getattr(args, "openrouter_site_url", ""),
            "X-Title": getattr(args, "openrouter_app_name", "EconGym"),
        }
        options["default_headers"] = {key: value for key, value in headers.items() if value}
    return OpenAI(**options)
