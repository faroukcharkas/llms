from types import UnionType
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum

class PartType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    REASONING = "reasoning"
    TOOL_CALL = "tool-call"
    TOOL_RESULT = "tool-result"

class TextPart(BaseModel):
    type: PartType = PartType.TEXT
    text: str
    provider_options: dict[str, Any]


class ImagePart(BaseModel):
    type: PartType = PartType.IMAGE
    image: str
    media_type: str | None
    provider_options: dict[str, Any]

class FilePart(BaseModel):
    type: PartType = PartType.FILE
    data: str
    filename: str | None = None
    media_type: str
    provider_options: dict[str, Any]


class ReasoningPart(BaseModel):
    type: PartType = PartType.REASONING
    text: str
    provider_options: dict[str, Any]
    
    


class ToolCallPart(BaseModel):
    type: PartType = PartType.TOOL_CALL
    tool_call_id: str
    tool_name: str
    input: Any
    provider_options: dict[str, Any]
    provider_executed: bool | None


class ToolResultPart(BaseModel):
    type: PartType = PartType.TOOL_RESULT
    tool_call_id: str
    tool_name: str
    output: Any
    provider_options: dict[str, Any]
    provider_executed: bool | None


ContentPart: UnionType = TextPart | ImagePart | FilePart | ReasoningPart | ToolCallPart | ToolResultPart

