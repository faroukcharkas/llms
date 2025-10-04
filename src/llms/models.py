from llms.types.enums import Provider

MODEL_MAP: dict[str, str] = {
    "gpt-4o": Provider.OPENAI,
    "gpt-5": Provider.OPENAI,
    "claude-sonnet-4-5": Provider.ANTHROPIC,
    "deepseek-r1": Provider.FIREWORKS,
    "llama-v3p1-8b-instruct": Provider.FIREWORKS,
    "gpt-oss-120b": Provider.FIREWORKS,
}

