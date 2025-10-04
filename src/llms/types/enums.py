from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FIREWORKS = "fireworks"

class Role(str, Enum): 
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"