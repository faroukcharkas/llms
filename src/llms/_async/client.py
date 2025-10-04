from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from llms.types.messages import ModelMessage
from llms.types.results import GenerateTextResult
from llms.models import MODEL_MAP
from llms.types.enums import Provider
from llms._async.handlers import handle_openai_generate_text

class AsyncLLM():
    openai_client: AsyncOpenAI
    anthropic_client: AsyncAnthropic
    fireworks_client: AsyncOpenAI


    def __init__(self, openai_key: str | None = None, anthropic_key: str | None = None, fireworks_key: str | None = None):
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
        self.fireworks_client = AsyncOpenAI(api_key=fireworks_key, base_url="https://api.fireworks.ai/inference/v1")

    async def generate_text(self, model_name: str, messages: list[ModelMessage]) -> GenerateTextResult:
        assert model_name in MODEL_MAP, f"Model {model_name} not found"
        match MODEL_MAP[model_name]:
            case Provider.OPENAI:
                return await handle_openai_generate_text(openai_client=self.openai_client, model_name=model_name, messages=messages)
            case Provider.ANTHROPIC:
                return await self.anthropic_client.messages.create(model=model_name, messages=messages)
            case Provider.FIREWORKS:
                return await self.fireworks_client.chat.completions.create(model=model_name, messages=messages)
            case _:
                raise ValueError("Did not recognize LLM model name")