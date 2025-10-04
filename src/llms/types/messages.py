from pydantic import BaseModel
from llms.types.enums import Role
from llms.types.parts import TextPart, ImagePart, FilePart, ReasoningPart, ToolCallPart, ToolResultPart


class ModelMessage(BaseModel):
    role: Role
    content: str

class SystemModelMessage(ModelMessage):
    role: Role = Role.SYSTEM
    content: str
    

class UserModelMessage(ModelMessage):
    role: Role = Role.USER
    content: str | list[TextPart | ImagePart | FilePart]


class AssistantModelMessage(ModelMessage):
    role: Role = Role.ASSISTANT
    content: str | list[TextPart | ImagePart | FilePart | ReasoningPart | ToolCallPart | ToolResultPart]