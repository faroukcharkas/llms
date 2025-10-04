from llms._sync.client import SyncLLM
from llms.types.results import GenerateTextResult
from llms.types.messages import ModelMessage
from llms.types.enums import Role

def test_client():
    client: SyncLLM = SyncLLM(
        openai_key="<OPENAI_API_KEY>",
        anthropic_key="<ANTHROPIC_API_KEY>",
        fireworks_key="<FIREWORKS_API_KEY>"
    )
    result: GenerateTextResult = client.generate_text(model_name="gpt-oss-120b", messages=[ModelMessage(role=Role.USER, content="Hello, how are you?")])
    print(result)
    assert 0 == 1