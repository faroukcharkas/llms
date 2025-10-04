from pydantic import BaseModel
from llms.types.parts import ContentPart


class GenerateTextResult(BaseModel):
    text: str
    parts: list[ContentPart]