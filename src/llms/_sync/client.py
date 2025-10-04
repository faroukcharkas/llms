from openai import OpenAI
from anthropic import Anthropic
from llms.types.messages import ModelMessage
from llms.types.results import GenerateTextResult
from llms.models import MODEL_MAP
from llms.types.enums import Provider
from llms._sync.handlers import handle_openai_generate_text, handle_anthropic_generate_text, handle_fireworks_generate_text

class SyncLLM():
    openai_client: OpenAI
    anthropic_client: Anthropic
    fireworks_client: OpenAI


    def __init__(self, openai_key: str | None = None, anthropic_key: str | None = None, fireworks_key: str | None = None):
        self.openai_client = OpenAI(api_key=openai_key)
        self.anthropic_client = Anthropic(api_key=anthropic_key)
        self.fireworks_client = OpenAI(api_key=fireworks_key, base_url="https://api.fireworks.ai/inference/v1")

    def generate_text(self, model_name: str, messages: list[ModelMessage]) -> GenerateTextResult:
        assert model_name in MODEL_MAP, f"Model {model_name} not found"
        match MODEL_MAP[model_name]:
            case Provider.OPENAI:
                return handle_openai_generate_text(openai_client=self.openai_client, model_name=model_name, messages=messages)
            case Provider.ANTHROPIC:
                return handle_anthropic_generate_text(anthropic_client=self.anthropic_client, model_name=model_name, messages=messages)
            case Provider.FIREWORKS:
                return handle_fireworks_generate_text(fireworks_client=self.fireworks_client, model_name=model_name, messages=messages)
            case _:
                raise ValueError("Did not recognize LLM model name")
